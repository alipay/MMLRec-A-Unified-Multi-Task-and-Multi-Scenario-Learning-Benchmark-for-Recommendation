# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel


class ESCM(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(ESCM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)
        self.config = config
        self.data_config = config['data_config']
        self.model_config = config['model_config']

        self.model_name = self.model_config.get("model_name", "escm")
        self.task_names = self.model_config.get("task_names", ["ctr", "ctcvr"])
        self.task_types = self.model_config.get("task_types", ["binary", "binary"])
        self.num_experts = self.model_config.get("num_experts", 4)
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = self.model_config.get("expert_dnn_hidden_units", [256, 128])
        self.gate_dnn_hidden_units = self.model_config.get("gate_dnn_hidden_units", [64])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)

        self.counterfactual_w = 0.1
        self.global_w = 1

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

        self.ctr_dnn = DNN(input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.cvr_dnn = DNN(input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

        self.ctr_dnn_final_layer = nn.Linear(self.expert_dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_final_layer = nn.Linear(self.expert_dnn_hidden_units[-1], 1, bias=False)

        if self.model_name == 'escm_dr':
            self.imp_dnn = DNN(input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.imp_dnn_final_layer = nn.Linear(self.expert_dnn_hidden_units[-1], 1, bias=False)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.ctr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cvr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.ctr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.cvr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        ctr_output = self.ctr_dnn(dnn_input)
        cvr_output = self.cvr_dnn(dnn_input)

        ctr_logit = self.ctr_dnn_final_layer(ctr_output)
        cvr_logit = self.cvr_dnn_final_layer(cvr_output)

        ctr_pred = self.out(ctr_logit)
        cvr_pred = self.out(cvr_logit)

        ctcvr_pred = ctr_pred * cvr_pred  # CTCVR = CTR * CVR
        task_outs = torch.cat([ctr_pred, cvr_pred, ctcvr_pred], -1)

        if self.model_name == "escm_dr":
            imp_output = self.imp_dnn(dnn_input)
            imp_logit = self.imp_dnn_final_layer(imp_output)
            imp_pred = self.out(imp_logit)
            task_outs = torch.cat([ctr_pred, cvr_pred, ctcvr_pred, imp_pred], -1)

        return task_outs

    def counterfact_ipw(self, loss_cvr, ctr_num, o, ctr_out_one):
        ps = torch.multiply(ctr_out_one, ctr_num.float())
        min_v = torch.full_like(ps, 0.000001)
        ps = torch.maximum(ps, min_v)
        ips = torch.reciprocal(ps)
        batch_shape = torch.full_like(o, 1)
        batch_size = torch.sum(batch_shape.float(), 0)
        ips = torch.clip(ips, min=-15, max=15)
        ips = torch.multiply(ips, batch_size)
        ips.stop_gradient = True
        loss_cvr = torch.multiply(loss_cvr, ips)
        loss_cvr = torch.multiply(loss_cvr, o)
        return torch.mean(loss_cvr)

