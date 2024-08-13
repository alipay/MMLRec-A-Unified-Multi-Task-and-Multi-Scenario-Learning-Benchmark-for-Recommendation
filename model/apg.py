import torch
import torch.nn as nn
import math
import numpy as np

from .utils import DNN, PredictionLayer, combined_dnn_input, SharedSpecificLinear, activation_layer, DomainBatchNorm
from .basemodel import BaseModel

class APGLayer(nn.Module):
    def __init__(self, input_dim, output_dim, scene_emb_dim, activation='relu', generate_activation=None, inner_activation=None,
                       use_uv_shared=True,  mf_k=16, use_mf_p=True,  mf_p=4, device='cpu'):
        super(APGLayer, self).__init__()
        self.device = device
        self.use_uv_shared = use_uv_shared
        self.use_mf_p = use_mf_p
        self.input_dim = input_dim
        self.output_dim = output_dim
        if activation is not None:
            self.activation = activation_layer(activation)
        else:
            self.activation = None
        if inner_activation is not None:
            self.inner_activation = activation_layer(inner_activation)
        else:
            self.inner_activation = None

        min_dim = min(int(input_dim), int(output_dim))
        p_dim = math.ceil(float(min_dim) / float(mf_p))
        k_dim = math.ceil(float(min_dim) / float(mf_k))
        self.p_dim = p_dim
        self.k_dim = k_dim
        kk_weight_shape = [k_dim, k_dim]
        kk_weight_size = np.prod(kk_weight_shape)
        self.specific_weight_kk = DNN(inputs_dim=scene_emb_dim, hidden_units=[kk_weight_size],
                                      activation=generate_activation, device=device)
        self.specific_bias_kk = DNN(inputs_dim=scene_emb_dim, hidden_units=[kk_weight_shape[-1]],
                                    activation=generate_activation, device=device)

        if use_uv_shared:
            if use_mf_p:
                # 模块输入
                # n*p模块共享参数,增大共享参数规模，提高共享参数部分表征能力
                self.shared_weight_np = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, p_dim), device=device)))
                self.shared_bias_np = nn.Parameter(nn.init.zeros_(torch.empty((p_dim,), device=device)))
                self.shared_weight_pk = nn.Parameter(nn.init.xavier_uniform_(torch.empty((p_dim, k_dim), device=device)))
                self.shared_bias_pk = nn.Parameter(nn.init.zeros_(torch.empty((k_dim,), device=device)))

                # 模块输出
                self.shared_weight_kp = nn.Parameter(nn.init.xavier_uniform_(torch.empty((k_dim, p_dim), device=device)))
                self.shared_bias_kp = nn.Parameter(nn.init.zeros_(torch.empty((p_dim,), device=device)))
                self.shared_weight_pm = nn.Parameter(nn.init.xavier_uniform_(torch.empty((p_dim, output_dim), device=device)))
                self.shared_bias_pm = nn.Parameter(nn.init.zeros_(torch.empty((output_dim,), device=device)))
            else:
                # 模块输入
                self.shared_weight_nk = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, k_dim), device=device)))
                self.shared_bias_nk = nn.Parameter(nn.init.zeros_(torch.empty((k_dim,), device=device)))

                # 模块输出
                self.shared_weight_km = nn.Parameter(nn.init.xavier_uniform_(torch.empty((k_dim, output_dim), device=device)))
                self.shared_bias_km = nn.Parameter(nn.init.zeros_(torch.empty((output_dim,), device=device)))
        else:
            # UV模块整体不共享的话，默认不需要进行P模块的过渡
            nk_weight_shape = [input_dim, k_dim]
            km_weight_shape = [k_dim, output_dim]
            nk_weight_size = np.prod(nk_weight_shape)
            km_weight_size = np.prod(km_weight_shape)
            # 模块输入
            self.specific_weight_nk = DNN(inputs_dim=scene_emb_dim, hidden_units=[nk_weight_size],
                                     activation=generate_activation, device=device)
            self.specific_bias_nk = DNN(inputs_dim=scene_emb_dim, hidden_units=[nk_weight_shape[-1]],
                                   activation=generate_activation, device=device)
            self.specific_weight_km = DNN(inputs_dim=scene_emb_dim, hidden_units=[km_weight_size],
                                     activation=generate_activation, device=device)
            self.specific_bias_km = DNN(inputs_dim=scene_emb_dim, hidden_units=[km_weight_shape[-1]],
                                   activation=generate_activation, device=device)
        self.to(device)
    def forward(self, inputs, scene_emb):
        specific_weight_kk = self.specific_weight_kk(scene_emb)
        specific_weight_kk = specific_weight_kk.view(-1, self.k_dim, self.k_dim)
        specific_bias_kk = self.specific_bias_kk(scene_emb)
        if self.use_uv_shared:
            if self.use_mf_p:
                output_np = torch.matmul(inputs, self.shared_weight_np) + self.shared_bias_np
                if self.inner_activation is not None:
                    output_np = self.inner_activation(output_np)
                output_pk = torch.matmul(output_np, self.shared_weight_pk) + self.shared_bias_pk
                if self.inner_activation is not None:
                    output_pk = self.inner_activation(output_pk)
                output_kk = torch.matmul(output_pk.unsqueeze(1), specific_weight_kk).squeeze() + specific_bias_kk
                if self.inner_activation is not None:
                    output_kk = self.inner_activation(output_kk)
                output_kp = torch.matmul(output_kk, self.shared_weight_kp) + self.shared_bias_kp
                if self.inner_activation is not None:
                    output_kp = self.inner_activation(output_kp)
                output_pm = torch.matmul(output_kp, self.shared_weight_pm) + self.shared_bias_pm
                output = output_pm
            else:
                output_nk = torch.matmul(inputs, self.shared_weight_nk) + self.shared_bias_nk
                if self.inner_activation is not None:
                    output_nk = self.inner_activation(output_nk)
                output_kk = torch.matmul(output_nk.unsqueeze(1), specific_weight_kk).squeeze() + specific_bias_kk
                if self.inner_activation is not None:
                    output_kk = self.inner_activation(output_kk)
                output_km = torch.matmul(output_kk, self.shared_weight_km) + self.shared_bias_km
                output = output_km
        else:
            specific_weight_nk = self.specific_weight_nk(scene_emb)
            specific_weight_nk = specific_weight_nk.view(-1, self.input_dim, self.k_dim)
            specific_bias_nk = self.specific_bias_nk(scene_emb)
            specific_weight_km = self.specific_weight_km(scene_emb)
            specific_weight_km = specific_weight_km.view(-1, self.k_dim, self.output_dim)
            specific_bias_km = self.specific_bias_km(scene_emb)
            output_nk = torch.matmul(inputs.unsqueeze(1), specific_weight_nk).squeeze() + specific_bias_nk
            if self.inner_activation is not None:
                output_nk = self.inner_activation(output_nk)
            output_kk = torch.matmul(output_nk.unsqueeze(1), specific_weight_kk).squeeze() + specific_bias_kk
            if self.inner_activation is not None:
                output_kk = self.inner_activation(output_kk)
            output_km = torch.matmul(output_kk.unsqueeze(1), specific_weight_km).squeeze() + specific_bias_km
            output = output_km

        if self.activation is not None:
            output = self.activation(output)
        return output



class APG(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(APG, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

        self.dnn_use_bn = self.model_config.get("dnn_use_bn", False)
        self.dnn_hidden_units = self.model_config.get("dnn_hidden_units", [256, 128])
        scene_emb_dim = self.model_config.get("emb", 8)
        scene_feature = self.data_config.get("scene_feature", "")
        dnn_activation = self.model_config.get("dnn_activation", "relu")

        if scene_feature != "":
            print(self.feature_index)
            self.scene_index = self.feature_index[scene_feature]

        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")

        input_dim = self.compute_input_dim(dnn_feature_columns)
        dnn_hidden_units = [input_dim] + self.dnn_hidden_units
        self.apg_layers = nn.ModuleList([APGLayer(input_dim=dnn_hidden_units[i], output_dim=dnn_hidden_units[i+1],
                                                  scene_emb_dim=scene_emb_dim, activation=dnn_activation,
                                                  use_uv_shared=True, use_mf_p=False, mf_k=4, mf_p=4,
                                                  device=device)
                                         for i in range(len(self.dnn_hidden_units))])
        self.final_layer = nn.ModuleList([nn.Linear(self.dnn_hidden_units[-1] , 1, bias=False) for _ in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        self.dnn_input = dnn_input
        scene_emb = sparse_embedding_list[self.scene_index[0]]
        scene_emb = scene_emb.squeeze().detach()
        apg_ouput = dnn_input
        apg_ouputs = []
        for i in range(len(self.dnn_hidden_units)):
            apg_ouput = self.apg_layers[i](apg_ouput, scene_emb)
            apg_ouputs.append(apg_ouput)
        self.last_layer = apg_ouput
        # apg_ouput= self.final_layer(apg_ouput)
        # print(f"output_size:{output.size()}")

        task_outs = []
        for i in range(self.num_tasks):
            apg_final_ouput = self.final_layer[i](apg_ouput)
            output = self.out[i](apg_final_ouput)
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
            num_layer = len(self.dnn_hidden_units)
            for i in range(num_layer):
                layer_output_dict[f'apg_output_{i}'] = apg_ouputs[i]
            self.layer_output_dict = layer_output_dict
        return task_outs
