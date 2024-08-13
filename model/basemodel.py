# -*- coding:utf-8 -*-
from __future__ import print_function

import time
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
from .utils import *
from .optimizer import PCGrad


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, init_std=0.0001,
                 device='cpu', gpus=None, config=None):

        super(BaseModel, self).__init__()
        # torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.config = config
        self.data_config = config['data_config']
        self.model_config = config['model_config']
        self.optim_config = config["optim_config"]
        self.training_config = config["training_config"]
        self.save_layer_output = False
        self.use_cka_loss = self.model_config.get("use_cka_loss", False)

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        task = self.model_config.get("task", "binary")
        self.task_name = self.model_config.get("task_name", "mtl")
        self.task_names = self.model_config.get("task_names", ["ctr", "ctcvr"])
        self.task_types = self.model_config.get("task_types", ["binary", "binary"])
        self.num_domains = self.data_config.get("num_domains", 1)
        if self.task_name == "msl":
            self.num_tasks = self.num_domains
        elif self.task_name == "mtmsl":
            self.num_tasks = len(self.data_config['label_columns'])
        else:
            self.num_tasks = len(self.task_names)

        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1!")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(self.task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in self.task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

        l2_reg_linear = self.model_config.get("l2_reg_linear", 1e-5)
        l2_reg_embedding = self.model_config.get("l2_reg_embedding", 1e-5)

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        domain_mask, domain_mask_val = None, None
        if self.model_config.get('task_name', 'mtl') == 'msl' or 'mtmsl':
            mask_column = self.data_config.get('mask_column', '')
            mask_values = self.data_config.get('mask_values', [])
            num_domains = self.data_config.get('num_domains', 1)
            if mask_column != '':
                mask_x = x[mask_column].values
                domain_values = list(mask_x)
                domain_mask = get_mask(domain_values, mask_values, num_domains)

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        if len(y[0]) < len(x[0]):
            data_len = len(x[0])
            y = np.array(y).reshape((data_len, self.num_tasks))

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if self.model_config.get('task_name', 'mtl') == 'msl' or 'mtmsl':
                if mask_column != '':
                    mask_x_val = val_x[mask_column].values
                    domain_values_val = list(mask_x_val)
                    domain_mask_val = get_mask(domain_values_val, mask_values, num_domains)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            if self.model_config.get('task_name', 'mtl') == 'msl' or 'mtmsl' and domain_mask is not None:
                domain_mask, domain_mask_val = domain_mask[:split_at, :], domain_mask[split_at:, :]

        else:
            val_x = []
            val_y = []

        if len(val_y) > 0 and len(val_x) > 0 and len(val_y[0]) != len(val_x[0]):
            data_len = len(val_x[0])
            val_y = np.array(val_y).reshape((data_len, self.num_tasks))


        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        if domain_mask is None:
            domain_mask = torch.ones([len(x[0]), 1])
        # print(domain_mask.size())
        # print(len(x[0]))
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            domain_mask,
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        best_auc = 0
        early_stop = 0
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}

            # for (x_train, domain_mask, y_train) in tqdm(train_loader):
            for (x_train, domain_mask, y_train) in train_loader:
                x = x_train.to(self.device).float()
                domain_mask = domain_mask.to(self.device).float()
                y = y_train.to(self.device).float()
                if self.model_config.get('task_name', 'mtl') != 'msl' or self.model_config.get('task_name', 'mtl') != 'mtmsl':
                    domain_mask = None

                y_pred = model(x, domain_mask).squeeze()
                optim.zero_grad()
                if isinstance(loss_func, list):
                    assert len(loss_func) == self.num_tasks,\
                        "the length of `loss_func` should be equal with `self.num_tasks`"
                    if domain_mask is not None:
                        if self.model_config.get('task_name', 'mtl') == 'msl':
                            loss = sum([loss_func[i](y_pred[:, i], y[:, i], weight=domain_mask[:, i], reduction='sum') for i in range(self.num_tasks)])
                        elif self.model_config.get('task_name', 'mtl') == 'mtmsl':
                            tmp = []
                            for i in range(self.num_tasks):
                                l = self.num_domains
                                j = i % l
                                tmp.append(loss_func[i](y_pred[:, i], y[:, i], weight=domain_mask[:, j], reduction='sum'))
                            loss = sum(tmp)
                    else:
                        if self.model_config['model_name'] == 'escm':
                            loss_0 = loss_func[0](y_pred[:, 0], y[:, 0], reduction='sum')
                            loss_1 = loss_func[1](y_pred[:, 1], y[:, 1], reduction='sum')
                            loss_2 = loss_func[1](y_pred[:, 2], y[:, 1], reduction='sum')

                            ctr_num = torch.sum(y[:, 0])
                            o = y[:, 0].float()
                            loss_1 = model.counterfact_ipw(loss_1, ctr_num, o, y_pred[:, 0])
                            loss = loss_0 + loss_1 * model.counterfactual_w + loss_2 * model.global_w
                        else:
                            loss = sum(
                                [loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in
                                 range(self.num_tasks)])
                else:
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                reg_loss = self.get_regularization_loss()
                if self.model_config.get('task_name', 'mtl') in ['msl', 'mtmsl'] and self.use_cka_loss:
                    cka_loss = self.add_cka_loss(self.last_layer, domain_mask, alpha=0.5)
                else:
                    cka_loss = torch.zeros((1,), device=self.device)

                total_loss = loss + reg_loss + self.aux_loss + cka_loss

                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                if self.model_config['model_name'] == 'pcg':
                    optim.pc_backward(total_loss)
                else:
                    total_loss.backward()
                optim.step()

                # if verbose > 0:
                for name, metric_fun in self.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    if self.task_name == "msl":
                        train_result[name].append(metric_fun(
                            y[:, 0].cpu().data.numpy(), y_pred.sum(dim=-1).cpu().data.numpy().astype("float64")))
                    elif self.task_name == "mtmsl":
                        y_new = y[:, [0, self.num_domains]].cpu().data.numpy()
                        y_pred_new = torch.cat((y_pred[:, :self.num_domains].sum(dim=-1).unsqueeze(-1),
                                            y_pred[:, self.num_domains:].sum(dim=-1).unsqueeze(-1)), dim=-1).cpu().data.numpy()
                        train_result[name].append(metric_fun(y_new, y_pred_new))
                    else:
                        if self.model_config['model_name'] == 'escm':
                            y_pred = torch.cat([y_pred[:, 0].unsqueeze(1), y_pred[:, 2].unsqueeze(1)], -1)
                        train_result[name].append(metric_fun(
                            y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            epoch_logs["cka_loss"] = cka_loss.item()
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size, domain_mask_val)
                print(eval_result)
                if eval_result['auc'] > best_auc:
                    best_auc = eval_result['auc']
                    best_model = deepcopy(model)
                    early_stop = 0
                else:
                    early_stop += 1
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - start_time)
            print('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s - loss: {1: .4f} - cka_loss: {2: .4f}".format(
                epoch_time, epoch_logs["loss"], epoch_logs["cka_loss"])

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            print(eval_str)
            # if self.stop_training:
            #     break
            if early_stop >= self.optim_config.get("early_stop", 3):
                break

        return best_model

    def evaluate(self, x, y, batch_size=256, domain_mask=None):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size, domain_mask)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if self.task_name == "msl":
                eval_result[name] = metric_fun(y[:, 0], np.sum(pred_ans, axis=-1))
            elif self.task_name == "mtmsl":
                y_new = y[:, [0, self.num_domains]]
                y_pred = np.concatenate((np.sum(pred_ans[:, :self.num_domains], axis=-1)[:, np.newaxis],
                                        np.sum(pred_ans[:, self.num_domains:], axis=-1)[:, np.newaxis]), axis=-1)
                eval_result[name] = metric_fun(y_new, y_pred)
            else:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256, domain_mask=None):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if self.model_config.get('task_name', 'mtl') == 'msl' or 'mtmsl' and domain_mask is None:
            mask_column = self.data_config.get('mask_column', '')
            mask_values = self.data_config.get('mask_values', [])
            num_domains = self.data_config.get('num_domains', 1)
            if mask_column != '':
                mask_x = x[mask_column].values
                domain_values = list(mask_x)
                domain_mask = get_mask(domain_values, mask_values, num_domains)

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        if domain_mask is None:
            domain_mask = torch.ones([len(x[0]), 1])

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            domain_mask
            )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []

        if not self.training and self.save_layer_output:
            layer_output_dict = {}
        with torch.no_grad():
            for x_test, domain_mask in test_loader:
                x = x_test.to(self.device).float()

                domain_mask = domain_mask.to(self.device).float()
                if self.model_config.get('task_name', 'mtl') != 'msl' or self.model_config.get('task_name', 'mtl') != 'mtmsl':
                    domain_mask = None
                if self.model_config['model_name'] == 'escm':
                    y_pred = model(x, domain_mask)
                    y_pred = torch.cat([y_pred[:, 0].unsqueeze(1), y_pred[:, 2].unsqueeze(1)], -1)
                    y_pred = y_pred.cpu().data.numpy()
                else:
                    y_pred = model(x, domain_mask).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
                if not self.training and self.save_layer_output:
                    for key, val in self.layer_output_dict.items():
                        if key not in layer_output_dict:
                            layer_output_dict[key] = []
                        layer_output_dict[key].append(val.cpu().data.numpy())
        print(f"save:{self.save_layer_output}")
        print(f"train: {self.training}")
        if not self.training and self.save_layer_output:
            for key, val in layer_output_dict.items():
                layer_output_dict[key] = np.concatenate(val).astype("float64")
            return np.concatenate(pred_ans).astype("float64"), layer_output_dict
        else:
            return np.concatenate(pred_ans).astype("float64")
    def update_save(self, value=True):
        self.save_layer_output = value

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim
    def compute_sparse_feat(self, feature_columns):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        return sparse_feature_columns

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def add_cka_loss(self, dnn_input, domain_mask, alpha=0.01):
        cka_loss = torch.zeros((1,), device=self.device)
        if self.model_config.get('task_name', 'mtl') == 'msl' and self.use_cka_loss:
            from utils.CKA import linear_CKA_torch
            for i in range(self.num_tasks - 1):
                for j in range(i + 1, self.num_tasks):
                    emb_i = dnn_input * domain_mask[:, i].view(-1, 1)
                    emb_j = dnn_input * domain_mask[:, j].view(-1, 1)
                    cka_ij = linear_CKA_torch(emb_i.T, emb_j.T, device=self.device)
                    cka_loss += cka_ij
        return cka_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        # self.args = args
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        if self.model_config['model_name'] == 'pcg':
            self.optim = PCGrad(self.optim)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        lr = self.optim_config.get("lr", 1e-3)
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_single) for loss_single in loss]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
