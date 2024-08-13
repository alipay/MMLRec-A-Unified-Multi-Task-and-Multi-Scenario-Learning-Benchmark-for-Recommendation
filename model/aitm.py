import torch
import torch.nn as nn
import numpy as np
from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel


class AITM(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(AITM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)
        self.config = config
        self.data_config = config['data_config']
        self.model_config = config['model_config']

        self.task_names = self.model_config.get("task_names", ["ctr", "ctcvr"])
        self.task_types = self.model_config.get("task_types", ["binary", "binary"])
        self.num_experts = self.model_config.get("num_experts", 4)
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.bottom_dnn_hidden_units = self.model_config.get("expert_dnn_hidden_units", [256, 128])
        self.gate_dnn_hidden_units = self.model_config.get("gate_dnn_hidden_units", [64])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)

        self.num_tasks = len(self.task_names)
        if self.num_tasks != 2:
            raise ValueError("the length of task_names must be equal to 2")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(self.task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in self.task_types:
            if task_type != 'binary':
                raise ValueError("task must be binary in ESMM, {} is illegal".format(task_type))

        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.g = torch.nn.ModuleList(
            [torch.nn.Linear(self.bottom_dnn_hidden_units[-1], self.bottom_dnn_hidden_units[-1]) for _ in
             range(self.num_tasks - 1)])
        self.h1 = torch.nn.Linear(self.bottom_dnn_hidden_units[-1], self.bottom_dnn_hidden_units[-1])
        self.h2 = torch.nn.Linear(self.bottom_dnn_hidden_units[-1], self.bottom_dnn_hidden_units[-1])
        self.h3 = torch.nn.Linear(self.bottom_dnn_hidden_units[-1], self.bottom_dnn_hidden_units[-1])

        self.bottom = torch.nn.ModuleList([DNN(self.input_dim, self.bottom_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])

        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.bottom_dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)

        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else self.bottom_dnn_hidden_units[
                -1], 1,
            bias=False) for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bottom.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn_final_layer.named_parameters()),
            l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        feat = [self.bottom[i](dnn_input) for i in range(self.num_tasks)]

        for i in range(1, self.num_tasks):
            p = self.g[i - 1](feat[i - 1]).unsqueeze(1)
            q = feat[i].unsqueeze(1)
            x = torch.cat([p, q], dim=1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            feat[i] = torch.sum(
                torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.bottom_dnn_hidden_units[-1]),
                                            dim=1) * V, 1)

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](feat[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](feat[i])
            output = self.out[i](tower_dnn_logit)
            if self.task_name == "msl" and domain_mask is not None:
                output = output * domain_mask[:, i]
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)

        return task_outs

