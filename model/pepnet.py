import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input, SharedSpecificLinear, activation_layer, DomainBatchNorm
from .basemodel import BaseModel


class GateNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="relu",
                 dropout_rate=0.0,
                 batch_norm=False,
                 device='cpu'):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(activation_layer(hidden_activation))
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)
        self.to(device)

    def forward(self, inputs):
        return self.gate(inputs) * 2

class PPNetBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 gate_input_dim=32,
                 gate_hidden_dim=None,
                 hidden_units=[],
                 hidden_activations="relu",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True,
                 device='cpu'):
        super(PPNetBlock, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [activation_layer(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers = []
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.gate_layers.append(GateNN(gate_input_dim, gate_hidden_dim,
                                           output_dim=hidden_units[idx]))
            self.mlp_layers.append(nn.Sequential(*dense_layers))
        self.gate_layers.append(GateNN(gate_input_dim, gate_hidden_dim,
                                       output_dim=hidden_units[-1]))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        self.to(device)

    def forward(self, feature_emb, gate_emb):
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        hidden = feature_emb
        for i in range(len(self.mlp_layers)):
            gw = self.gate_layers[i](gate_input)
            hidden = self.mlp_layers[i](hidden * gw)
        return hidden


class PepNet(BaseModel):

    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(PepNet, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

        self.dnn_use_bn = self.model_config.get("dnn_use_bn", False)
        self.dnn_hidden_units = self.model_config.get("dnn_hidden_units", [256, 128])
        scene_emb_dim = self.model_config.get("emb", 8)
        scene_feature = self.data_config.get("scene_feature", "")
        self.user_sf = self.data_config.get("user_sf", "")
        self.item_sf = self.data_config.get("item_sf", "")

        task_dim = 0
        if scene_feature != "":
            # print(self.feature_index)
            self.scene_index = self.feature_index[scene_feature]
            task_dim += self.model_config.get("emb", 8)
        if self.user_sf != "":
            self.user_index = self.feature_index[self.user_sf]
            task_dim += self.model_config.get("emb", 8)
        if self.item_sf != "":
            self.item_index = self.feature_index[self.item_sf]
            task_dim += self.model_config.get("emb", 8)

        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")

        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.feature_gate = GateNN(input_dim=input_dim+scene_emb_dim, hidden_dim=128, output_dim=input_dim, device=device)
        self.ppn = nn.ModuleList([PPNetBlock(input_dim=input_dim,
                              output_dim=1,
                              gate_input_dim=input_dim+task_dim,
                              gate_hidden_dim=None,
                              hidden_units=self.dnn_hidden_units,
                              device=device) for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        self.dnn_input = dnn_input
        scene_emb = sparse_embedding_list[self.scene_index[0]]
        scene_emb = scene_emb.squeeze().detach()
        if self.user_sf != "":
            user_sf_emb = sparse_embedding_list[self.user_index[0]]
            user_sf_emb = user_sf_emb.squeeze().detach()
        if self.item_sf != "":
            item_sf_emb = sparse_embedding_list[self.item_index[0]]
            item_sf_emb = item_sf_emb.squeeze().detach()
        if self.user_sf != "" and self.item_sf != "":
            task_sf_emb = torch.cat([scene_emb, user_sf_emb, item_sf_emb],dim=-1)
        else:
            task_sf_emb = scene_emb

        feature_gate = self.feature_gate(torch.cat([dnn_input.detach(), scene_emb], dim=-1))
        dnn_input = feature_gate * dnn_input
        pepnet_outputs = []
        for i in range(self.num_tasks):
            pepnet_output = self.ppn[i](dnn_input, task_sf_emb)
            pepnet_outputs.append(pepnet_output)

        task_outs = []
        for i in range(self.num_tasks):
            output = self.out[i](pepnet_outputs[i])
            if self.task_name == "msl" and domain_mask is not None:
                output = output * domain_mask[:, i].view(-1, 1)
            elif self.task_name == "mtmsl" and domain_mask is not None:
                l = self.num_domains
                j = i % l
                output = output * domain_mask[:, j].view(-1, 1)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs