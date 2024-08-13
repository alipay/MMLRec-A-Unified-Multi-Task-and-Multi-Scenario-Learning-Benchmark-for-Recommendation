import torch
import torch.nn as nn
from .utils import DNN, PredictionLayer, combined_dnn_input, activation_layer
from .basemodel import BaseModel


class CrossStitchLayer(nn.Module):
    def __init__(self, input_dims, device='cpu'):
        super(CrossStitchLayer, self).__init__()
        self.last_dims = input_dims
        self.total_last_dim = sum(self.last_dims)
        self.cross_stitch_weight = nn.Parameter(nn.init.eye_(torch.empty((self.total_last_dim, self.total_last_dim), device=device)))

    def forward(self, inputs):
        if not isinstance(inputs, (list or tuple)) or len(inputs) <= 1:
            print('inputs is not list or tuple, nothing to cross stitch')
            return inputs
        combined_input = torch.cat(inputs, dim=-1)
        cross_stitch_output = torch.matmul(combined_input, self.cross_stitch_weight)
        outputs = []
        end_index = 0
        for i in range(len(inputs)):
            start_index = end_index
            end_index += self.last_dims[i]
            shape_i = inputs[i].shape[-1]
            outputs.append(cross_stitch_output[:, start_index:end_index].reshape((-1, shape_i)))
        return outputs


class CrossStitch(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(CrossStitch, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)
        self.shared_hidden_unit = self.model_config.get("shared_hidden_unit", 256)
        self.dnn_hidden_units = self.model_config.get("dnn_hidden_units", [256, 128])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)
        self.device = device


        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.input_dim = input_dim
        self.shared_layer = DNN(self.input_dim, [self.shared_hidden_unit], activation=dnn_activation,
                                 l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                 init_std=init_std, device=device)
        self.cross_stitch = nn.ModuleDict(
            {'task_layer_0': nn.ModuleList([DNN(self.shared_hidden_unit, [self.dnn_hidden_units[0]], activation=dnn_activation,
                                           l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                           init_std=init_std, device=device) for _ in range(self.num_tasks)])})
        self.cross_stitch['gate_0'] = CrossStitchLayer(input_dims=[self.dnn_hidden_units[0]] * self.num_tasks, device=device)
        for i in range(1, len(self.dnn_hidden_units)):
            self.cross_stitch[f'task_layer_{i}'] = nn.ModuleList([DNN(self.dnn_hidden_units[i - 1],
                                                                         [self.dnn_hidden_units[i]],
                                                                         activation=dnn_activation,
                                                                         l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                                                         use_bn=dnn_use_bn,
                                                                         init_std=init_std, device=device) for _ in
                                                                     range(self.num_tasks)])
            self.cross_stitch[f'gate_{i}'] = CrossStitchLayer(input_dims=[self.dnn_hidden_units[i]] * self.num_tasks, device=device)

        # tower dnn (task-specific)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else
            self.dnn_hidden_units[-1], 1, bias=False)
            for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # cross_stitch_dnn
        cross_stitch_outputs = [self.shared_layer(dnn_input)] * self.num_tasks
        for i in range(len(self.dnn_hidden_units)):
            for j in range(self.num_tasks):
                cross_stitch_outputs[j] = self.cross_stitch[f'task_layer_{i}'][j](cross_stitch_outputs[j])
            cross_stitch_outputs = self.cross_stitch[f'gate_{i}'](cross_stitch_outputs)

        task_outs = []
        tower_dnn_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](cross_stitch_outputs[i])
                tower_dnn_outs.append(tower_dnn_out)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](cross_stitch_outputs[i])
            output = self.out[i](tower_dnn_logit)
            if self.task_name == "msl" and domain_mask is not None:
                output = output * domain_mask[:, i].view(-1, 1)
            elif self.task_name == "mtmsl" and domain_mask is not None:
                l = self.num_domains
                j = i % l
                output = output * domain_mask[:, j].view(-1, 1)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)

        if not self.training and self.save_layer_output:
            layer_output_dict = {}
            layer_output_dict['dnn_input'] = dnn_input
            layer_output_dict['cross_stitch_outputs'] = torch.stack(cross_stitch_outputs, 1)
            if len(self.tower_dnn_hidden_units) > 0:
                layer_output_dict['tower_outputs'] = torch.stack(tower_dnn_outs, 1)
            self.layer_output_dict = layer_output_dict

        return task_outs
