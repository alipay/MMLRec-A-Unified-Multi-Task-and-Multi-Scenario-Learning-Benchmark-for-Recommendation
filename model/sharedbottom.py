# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel

class SharedBottom(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
    
        super(SharedBottom, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                           init_std=init_std, device=device, gpus=gpus, config=config)

        self.num_experts = self.model_config.get("num_experts", 4)
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.bottom_dnn_hidden_units = self.model_config.get("bottom_dnn_hidden_units", [256, 128])
        self.gate_dnn_hidden_units = self.model_config.get("gate_dnn_hidden_units", [64])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)


        self.bottom_dnn = DNN(self.input_dim, self.bottom_dnn_hidden_units, activation=dnn_activation,
                              dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                              init_std=init_std, device=device)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.bottom_dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else self.bottom_dnn_hidden_units[-1], 1,
            bias=False) for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bottom_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn_final_layer.named_parameters()),
            l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        shared_bottom_output = self.bottom_dnn(dnn_input)

        # tower dnn (task-specific)
        task_outs = []
        tower_dnn_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](shared_bottom_output)
                tower_dnn_outs.append(tower_dnn_out)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](shared_bottom_output)
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
            layer_output_dict['shared_bottom_outputs'] = shared_bottom_output
            if len(self.tower_dnn_hidden_units) > 0:
                layer_output_dict['tower_outputs'] = torch.stack(tower_dnn_outs, 1)
            self.layer_output_dict = layer_output_dict

        return task_outs
