# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input
from .basemodel import BaseModel


class PLE(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(PLE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                 init_std=init_std, device=device, gpus=gpus, config=config)


        self.num_experts = self.model_config.get("num_experts", 4)
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.shared_expert_num = self.model_config.get("shared_expert_num", 1)
        self.specific_expert_num = self.model_config.get("specific_expert_num", 3)
        self.num_levels = self.model_config.get("num_levels", 1)
        self.expert_dnn_hidden_units = self.model_config.get("expert_dnn_hidden_units", [256, 128])
        self.gate_dnn_hidden_units = self.model_config.get("gate_dnn_hidden_units", [64])
        self.tower_dnn_hidden_units = self.model_config.get("tower_dnn_hidden_units", [64])
        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)
        l2_reg_dnn = self.model_config.get("l2_reg_dnn", 0)
        dnn_dropout = self.model_config.get("dnn_dropout", 0)
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        dnn_use_bn = self.model_config.get("dnn_use_bn", False)


        def multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units):
            return nn.ModuleList(
                [nn.ModuleList([nn.ModuleList([DNN(inputs_dim_level0 if level_num == 0 else inputs_dim_not_level0,
                                                   hidden_units, activation=dnn_activation,
                                                   l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                   init_std=init_std, device=device) for _ in
                                               range(expert_num)])
                                for _ in range(num_tasks)]) for level_num in range(num_level)])

        # 1. experts
        # task-specific experts
        self.specific_experts = multi_module_list(self.num_levels, self.num_tasks, self.specific_expert_num,
                                                  self.input_dim, self.expert_dnn_hidden_units[-1], self.expert_dnn_hidden_units)

        # shared experts
        self.shared_experts = multi_module_list(self.num_levels, 1, self.specific_expert_num,
                                                self.input_dim, self.expert_dnn_hidden_units[-1], self.expert_dnn_hidden_units)

        # 2. gates
        # gates for task-specific experts
        specific_gate_output_dim = self.specific_expert_num + self.shared_expert_num
        if len(self.gate_dnn_hidden_units) > 0:
            self.specific_gate_dnn = multi_module_list(self.num_levels, self.num_tasks, 1,
                                                       self.input_dim, self.expert_dnn_hidden_units[-1],
                                                       self.gate_dnn_hidden_units)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.specific_gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.specific_gate_dnn_final_layer = nn.ModuleList(
            [nn.ModuleList([nn.Linear(
                self.gate_dnn_hidden_units[-1] if len(self.gate_dnn_hidden_units) > 0 else self.input_dim if level_num == 0 else
                self.expert_dnn_hidden_units[-1], specific_gate_output_dim, bias=False)
                for _ in range(self.num_tasks)]) for level_num in range(self.num_levels)])

        # gates for shared experts
        shared_gate_output_dim = self.num_tasks * self.specific_expert_num + self.shared_expert_num
        if len(self.gate_dnn_hidden_units) > 0:
            self.shared_gate_dnn = nn.ModuleList([DNN(self.input_dim if level_num == 0 else self.expert_dnn_hidden_units[-1],
                                                      self.gate_dnn_hidden_units, activation=dnn_activation,
                                                      l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                      init_std=init_std, device=device) for level_num in
                                                  range(self.num_levels)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.shared_gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(
                self.gate_dnn_hidden_units[-1] if len(self.gate_dnn_hidden_units) > 0 else self.input_dim if level_num == 0 else
                self.expert_dnn_hidden_units[-1], shared_gate_output_dim, bias=False)
                for level_num in range(self.num_levels)])

        # 3. tower dnn (task-specific)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(self.expert_dnn_hidden_units[-1], self.tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            self.tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else self.expert_dnn_hidden_units[-1], 1,
            bias=False)
            for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])

        regularization_modules = [self.specific_experts, self.shared_experts, self.specific_gate_dnn_final_layer,
                                  self.shared_gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    # a single cgc Layer
    def cgc_net(self, inputs, level_num):
        # inputs: [task1, task2, ... taskn, shared task]

        # 1. experts
        # task-specific experts
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_experts[level_num][i][j](inputs[i])
                specific_expert_outputs.append(specific_expert_output)

        # shared experts
        shared_expert_outputs = []
        for k in range(self.shared_expert_num):
            shared_expert_output = self.shared_experts[level_num][0][k](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        # 2. gates
        # gates for task-specific experts
        cgc_outs = []
        for i in range(self.num_tasks):
            # concat task-specific expert and task-shared expert
            cur_experts_outputs = specific_expert_outputs[
                                  i * self.specific_expert_num:(i + 1) * self.specific_expert_num] + shared_expert_outputs
            cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

            # gate dnn
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.specific_gate_dnn[level_num][i][0](inputs[i])
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](gate_dnn_out)
            else:
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](inputs[i])
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
            cgc_outs.append(gate_mul_expert.squeeze())

        # gates for shared experts
        cur_experts_outputs = specific_expert_outputs + shared_expert_outputs
        cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

        if len(self.gate_dnn_hidden_units) > 0:
            gate_dnn_out = self.shared_gate_dnn[level_num](inputs[-1])
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](gate_dnn_out)
        else:
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](inputs[-1])
        gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
        cgc_outs.append(gate_mul_expert.squeeze())

        return cgc_outs

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # repeat `dnn_input` for several times to generate cgc input
        ple_inputs = [dnn_input] * (self.num_tasks + 1)  # [task1, task2, ... taskn, shared task]
        ple_outputs_list = []
        for i in range(self.num_levels):
            ple_outputs = self.cgc_net(inputs=ple_inputs, level_num=i)
            ple_inputs = ple_outputs
            ple_outputs_list.append(torch.stack(ple_outputs, 1))

        # tower dnn (task-specific)
        task_outs = []
        tower_dnn_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](ple_outputs[i])
                tower_dnn_outs.append(tower_dnn_out)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](ple_outputs[i])
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
            for i in range(self.num_levels):
                layer_output_dict[f'ple_output_{i}'] = ple_outputs_list[i]
            if len(self.tower_dnn_hidden_units) > 0:
                layer_output_dict['tower_outputs'] = torch.stack(tower_dnn_outs, 1)
            self.layer_output_dict = layer_output_dict

        return task_outs
