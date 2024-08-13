import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input, SharedSpecificLinear, activation_layer, DomainBatchNorm
from .basemodel import BaseModel


class MLP(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(MLP, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

        self.dnn_use_bn = self.model_config.get("dnn_use_bn", False)
        self.dnn_hidden_units = self.model_config.get("dnn_hidden_units", [256, 128])
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)

        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")

        input_dim = self.compute_input_dim(dnn_feature_columns)
        dnn_hidden_units = [input_dim] + self.dnn_hidden_units
        print(f"hidden_units:{dnn_hidden_units}")

        self.mlp_layers = nn.ModuleList([DNN(inputs_dim=dnn_hidden_units[i], hidden_units=[dnn_hidden_units[i+1]]
                                             ,activation='relu', l2_reg=l2_reg_dnn, device=device)
                                         for i in range(len(self.dnn_hidden_units))])

        self.final_layer = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False, device=device)
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp_layers.named_parameters()),
            l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        self.dnn_input = dnn_input

        mlp_ouput = dnn_input
        mlp_ouputs = []
        for i in range(len(self.dnn_hidden_units)):
            mlp_ouput = self.mlp_layers[i](mlp_ouput)
            mlp_ouputs.append(mlp_ouput)
        self.last_layer = mlp_ouput
        mlp_output = self.final_layer(mlp_ouput)

        task_outs = []
        for i in range(self.num_tasks):
            output = self.out[i](mlp_output)
            if self.task_name == "msl" and domain_mask is not None:
                output = output * domain_mask[:, i].view(-1, 1)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)

        if not self.training and self.save_layer_output:
            layer_output_dict = {}
            layer_output_dict['dnn_input'] = dnn_input
            num_layer = len(self.dnn_hidden_units)
            for i in range(num_layer):
                layer_output_dict[f'mlp_output_{i}'] = mlp_ouputs[i]
            self.layer_output_dict = layer_output_dict

        return task_outs
