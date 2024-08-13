import torch
import torch.nn as nn

from .utils import DNN, PredictionLayer, combined_dnn_input, SharedSpecificLinear, activation_layer, DomainBatchNorm
from .basemodel import BaseModel


class STAR(BaseModel):
    def __init__(self, dnn_feature_columns, init_std=0.0001, device='cpu', gpus=None, config=None):
        super(STAR, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   init_std=init_std, device=device, gpus=gpus, config=config)

        self.dnn_use_bn = self.model_config.get("dnn_use_bn", False)
        self.dnn_hidden_units = self.model_config.get("dnn_hidden_units", [256, 128])
        dnn_activation = self.model_config.get("dnn_activation", "relu")
        use_shared = self.model_config.get("use_shared", True)

        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")

        input_dim = self.compute_input_dim(dnn_feature_columns)
        hidden_units = [input_dim]
        for unit in self.dnn_hidden_units:
            hidden_units.append(unit)
        print(f"hidden_units:{hidden_units}")
        self.linears = nn.ModuleList(
            [SharedSpecificLinear(hidden_units[i], hidden_units[i + 1], self.num_tasks, use_shared=use_shared, device=device)
             for i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList(
            [activation_layer(dnn_activation)
             for _ in range(len(hidden_units) - 1)])
        if self.dnn_use_bn:
            self.domain_bn = DomainBatchNorm(num_features=hidden_units[1], num_domains=self.num_tasks, device=device)

        self.final_layers = nn.ModuleList([SharedSpecificLinear(hidden_units[-1], 1, self.num_tasks, use_shared=use_shared, device=device) for _ in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])
        self.to(device)

    def forward(self, X, domain_mask=None):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        star_outputs = []
        star_layers = []
        for i in range(self.num_tasks):
            output = dnn_input
            for j in range(len(self.dnn_hidden_units)):
                output = self.linears[j](output, i)
                output = self.activation_layers[j](output)
                if j == 0 and self.dnn_use_bn and domain_mask is not None:
                    output = self.domain_bn(output, domain_mask, self.training)
                star_layers.append(output)
            output = self.final_layers[i](output, i)
            star_outputs.append(output)
        self.last_layer = star_layers[-1]
        assert len(star_outputs) == len(self.task_types)
        task_outs = []
        for i, star_output in enumerate(star_outputs):
            output = self.out[i](star_output)
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
                star_layer_i = []
                for j in range(self.num_tasks):
                    star_layer_i.append(star_layers[i+j*num_layer])
                layer_output_dict[f'star_output_{i}'] = torch.stack(star_layer_i, 1)
            self.layer_output_dict = layer_output_dict

        return task_outs
