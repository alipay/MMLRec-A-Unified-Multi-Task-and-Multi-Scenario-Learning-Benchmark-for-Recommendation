# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel


class HMOE(BaseModel):

    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(HMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

        self.num_experts = self.model_config.get("num_experts", 4)
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = self.model_config.get("expert_dnn_hidden_units", [256, 128])
        self.gate_dnn_hidden_units = self.model_config.get("gate_dnn_hidden_units", [64])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        self.task_weight_hidden_units = self.model_config.get("task_weight_hidden_units", [64])
        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)

        # expert dnn
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn
        if len(self.gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, self.gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(self.gate_dnn_hidden_units[-1] if len(self.gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.expert_dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)

        # task weight
        if len(self.task_weight_hidden_units) > 0:
            self.task_weight = nn.ModuleList(
                [DNN(self.input_dim, self.task_weight_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.task_weight.named_parameters()),
                l2=l2_reg_dnn)
        self.task_weight_final_layer = nn.ModuleList(
            [nn.Linear(self.task_weight_hidden_units[-1] if len(self.task_weight_hidden_units) > 0 else self.input_dim,
                       self.num_tasks, bias=False) for _ in range(self.num_tasks)])

        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else self.expert_dnn_hidden_units[
                -1], 1,
            bias=False)
            for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.task_weight_final_layer,
                                  self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn
        mmoe_outs = []
        gate_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)
            gate_outs.append(gate_dnn_out.softmax(1))
            mmoe_outs.append(gate_mul_expert.squeeze())

        # task weight
        task_weights = []
        for i in range(self.num_tasks):
            if len(self.task_weight_hidden_units) > 0:
                task_weight_out = self.task_weight[i](dnn_input)
                task_weight_out = self.task_weight_final_layer[i](task_weight_out)
            else:
                task_weight_out = self.task_weight_final_layer[i](dnn_input)
            task_weights.append(task_weight_out.softmax(1))

        tower_dnn_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_outs.append(tower_dnn_out)
            else:
                tower_dnn_outs.append(mmoe_outs[i])

        task_outs = []
        for i in range(self.num_tasks):
            task_i_out = task_weights[i][:, i].view(-1, 1) * tower_dnn_outs[i]
            for j in range(self.num_tasks):
                if j == i:
                    continue
                task_i_out += task_weights[i][:, j].view(-1, 1) * tower_dnn_outs[j].detach()
            tower_dnn_logit = self.tower_dnn_final_layer[i](task_i_out)
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
            layer_output_dict['expert_outputs'] = expert_outs
            layer_output_dict['mmoe_outputs'] = torch.stack(mmoe_outs, 1)
            layer_output_dict['gate_outputs'] = torch.stack(gate_outs, 1)
            if len(self.tower_dnn_hidden_units) > 0:
                layer_output_dict['tower_outputs'] = torch.stack(tower_dnn_outs, 1)
            self.layer_output_dict = layer_output_dict
        return task_outs
