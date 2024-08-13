import torch
import torch.nn as nn
import math
from .utils import DNN, PredictionLayer, combined_dnn_input, activation_layer
from .basemodel import BaseModel
from .snr_trans import SNR_trans


class gate(nn.Module):
    def __init__(self, input_dim, output_dim, units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, device='cpu'):
        super(gate, self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,), device=device), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 1.1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.device = device
        self.e = 1e-8
        self.activation_layer = activation_layer(activation, output_dim, dice_dim)
        self.w = [[torch.empty(units, device=device) for j in range(self.input_dim)] for i in range(self.output_dim)]
        self.u = [[torch.nn.Parameter(torch.nn.init.uniform_(self.w[i][j], self.e, 1 - self.e),
                                      requires_grad=True) for j in range(self.input_dim)] for i in
                  range(self.output_dim)]

        self.w_matrix = [[torch.empty(units, units, device=device) for j in range(self.input_dim)] for i in
                         range(self.output_dim)]

        self.trans_matrix = [[torch.nn.Parameter(torch.nn.init.xavier_normal_(self.w_matrix[i][j]),
                                                 requires_grad=True) for j in range(self.input_dim)] for i in
                             range(self.output_dim)]

        self.to(device)

    def forward(self, x):
        # u = torch.nn.init.uniform_(self.w, self.e, 1 - self.e).to(self.device)
        self.s = [
            [torch.sigmoid(torch.log(self.u[i][j]) - torch.log(1 - self.u[i][j]) + torch.log(self.alpha) / self.beta)
             for j in range(self.input_dim)] for i in range(self.output_dim)]
        self.s_ = [[self.s[i][j] * (self.eplison - self.gamma) + self.gamma for j in range(self.input_dim)] for i in
                   range(self.output_dim)]
        self.z_params = [[(self.s_[i][j] > 0).float() * self.s_[i][j] for j in range(self.input_dim)] for i in
                         range(self.output_dim)]
        self.z_params = [
            [(self.z_params[i][j] > 1).float() + (self.z_params[i][j] <= 1).float() * self.z_params[i][j] for j in
             range(self.input_dim)] for i in range(self.output_dim)]

        gate_outs = []
        for i in range(self.output_dim):
            outputs = torch.stack(
                [torch.matmul(x[j], self.trans_matrix[i][j]) * self.z_params[i][j] for j in range(self.input_dim)], 1)
            outputs = outputs.sum(1)
            gate_outs.append(outputs)
        return gate_outs


class MSSM(BaseModel):
    # def __init__(self, config):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(MSSM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
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

        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")

        self.hidden_units = [self.input_dim] + list(self.expert_dnn_hidden_units)

        self.mssm = nn.ModuleDict(
            {'expert1': nn.ModuleList([DNN(self.input_dim, [self.expert_dnn_hidden_units[0]], activation=dnn_activation,
                                           l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                           init_std=init_std, device=device) for _ in range(self.num_experts)])})
        if len(self.expert_dnn_hidden_units) == 1:
            self.mssm['gate1'] = gate(self.num_experts, self.num_tasks, self.expert_dnn_hidden_units[-1],
                                      activation=dnn_activation,
                                      l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                      init_std=init_std, device=device)
        else:
            self.mssm['gate1'] = gate(self.num_experts, self.num_experts, self.expert_dnn_hidden_units[0],
                                      activation=dnn_activation,
                                      l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                      init_std=init_std, device=device)
            for i in range(1, len(self.expert_dnn_hidden_units)):
                self.mssm['expert{}'.format(i + 1)] = nn.ModuleList([DNN(self.expert_dnn_hidden_units[i - 1],
                                                                         [self.expert_dnn_hidden_units[i]],
                                                                         activation=dnn_activation,
                                                                         l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                                                         use_bn=dnn_use_bn,
                                                                         init_std=init_std, device=device) for _ in
                                                                     range(self.num_experts)])
                if i == len(self.expert_dnn_hidden_units) - 1:
                    self.mssm['gate{}'.format(i + 1)] = gate(self.num_experts, self.num_tasks,
                                                             self.expert_dnn_hidden_units[-1],
                                                             activation=dnn_activation,
                                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                                             use_bn=dnn_use_bn,
                                                             init_std=init_std, device=device)
                else:
                    self.mssm['gate{}'.format(i + 1)] = gate(self.num_experts, self.num_experts,
                                                             self.expert_dnn_hidden_units[i], activation=dnn_activation,
                                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                                             use_bn=dnn_use_bn,
                                                             init_std=init_std, device=device)

        # tower dnn (task-specific)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.expert_dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else self.expert_dnn_hidden_units[
                -1], 1,
            bias=False)
            for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn
        gate_outs = []
        for i in range(len(self.expert_dnn_hidden_units)):
            outs = []
            if i == 0:
                for j in range(self.num_experts):
                    expert_out = self.mssm['expert{}'.format(i + 1)][j](dnn_input)
                    outs.append(expert_out)
                # expert_outs = outs

            else:
                for j in range(self.num_experts):
                    expert_out = self.mssm['expert{}'.format(i + 1)][j](gate_outs[j])
                    outs.append(expert_out)
            gate_outs = self.mssm['gate{}'.format(i + 1)](outs)

        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](gate_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](gate_outs[i])
            output = self.out[i](tower_dnn_logit)
            if self.task_name == "msl" and domain_mask is not None:
                output = output * domain_mask[:, i].view(-1, 1)
            elif self.task_name == "mtmsl" and domain_mask is not None:
                l = self.num_domains
                j = i % l
                output = output * domain_mask[:, j].view(-1, 1)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)

        return task_outs