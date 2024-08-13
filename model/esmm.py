# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel


class ESMM(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(ESMM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

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

        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.ctr_dnn = DNN(input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.cvr_dnn = DNN(input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

        self.ctr_dnn_final_layer = nn.Linear(self.expert_dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_final_layer = nn.Linear(self.expert_dnn_hidden_units[-1], 1, bias=False)

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

        task_outs = torch.cat([ctr_pred, ctcvr_pred], -1)

        if not self.training and self.save_layer_output:
            layer_output_dict = {}
            layer_output_dict['dnn_input'] = dnn_input
            layer_output_dict['target0_output'] = ctr_output
            layer_output_dict['target1_output'] = cvr_output
            self.layer_output_dict = layer_output_dict
        return task_outs

