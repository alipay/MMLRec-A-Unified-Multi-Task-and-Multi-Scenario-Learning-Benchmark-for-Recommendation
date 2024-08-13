import torch
import random
import argparse
from utils.data_utils import *
from model.mmoe import MMOE
from model.esmm import ESMM
from model.sharedbottom import SharedBottom
from model.ple import PLE
from model.snr_trans import SNR_trans
from model.mssm import MSSM
from model.star import STAR
from model.apg import APG
from model.mlp import MLP
from model.cross_stitch import CrossStitch
from model.aitm import AITM
from model.escm import ESCM
from model.hmoe import HMOE
from model.pepnet import PepNet
import pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_model(model_name, df_columns=None, config=None):
    name = model_name.lower()
    if name == 'mmoe':
        return MMOE(df_columns, device=device, config=config)
    elif name == 'esmm':
        return ESMM(df_columns, device=device, config=config)
    elif name == 'sharedbottom':
        return SharedBottom(df_columns, device=device, config=config)
    elif name == 'ple':
        return PLE(df_columns, device=device, config=config)
    elif name == 'snr_trans':
        return SNR_trans(df_columns, device=device, config=config)
    elif name == 'mssm':
        return MSSM(df_columns, device=device, config=config)
    elif name == 'star':
        return STAR(df_columns, device=device, config=config)
    elif name == 'pcg':
        return MMOE(df_columns, device=device, config=config)
    elif name == 'apg':
        return APG(df_columns, device=device, config=config)
    elif name == 'mlp':
        return MLP(df_columns, device=device, config=config)
    elif name == 'cross_stitch':
        return CrossStitch(df_columns, device=device, config=config)
    elif name == 'aitm':
        return AITM(df_columns, device=device, config=config)
    elif name == 'escm':
        return ESCM(df_columns, device=device, config=config)
    elif name == 'hmoe':
        return HMOE(df_columns, device=device, config=config)
    elif name == 'pepnet':
        return PepNet(df_columns, device=device, config=config)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--run', type=bool, default=False)
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--config', type=str, default='')
parser.add_argument('--is_parallel', type=bool, default=False)
parser.add_argument('--device', default='cuda')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)
    seed_list = [0, 2, 4, 8]
    # seed_list = [args.seed]
    for seed in seed_list:
        print("seed:", seed)
        set_seed(seed)
        config = unserialize(args.config)
        print(config)
        data_config = config["data_config"]
        model_config = config["model_config"]
        optim_config = config["optim_config"]
        training_config = config["training_config"]
        save_config = config["save_config"]
        if args.run:
            config["model_config"]["model_name"] = args.model_name
            print(config["model_config"]["model_name"])
        model_name = model_config.get("model_name", "sharedbottom")
        target = list(set(data_config.get("label_columns", ["label"])))
        train_batch_size = training_config.get("train_batch_size", 4096)
        test_batch_size = training_config.get("test_batch_size", 4096)
        epochs = training_config.get("epochs", 10)

        train, test, test_mask, train_model_input, test_model_input, lf_columns, df_columns = ctrdataset(config)

        model = get_model(model_name, df_columns=df_columns, config=config)
        model.compile(optimizer=optim_config.get("optimizer", "adagrad"),
                      loss=optim_config.get("loss", ["binary_crossentropy", "binary_crossentropy"]),
                      metrics=optim_config.get("metrics", ["auc", "acc"])
                      )

        best_model = model.fit(train_model_input, train[target].values, batch_size=train_batch_size, epochs=epochs, validation_data=(test_model_input, test[target].values))
        if save_config.get("save_layer_output", False):
            best_model.update_save()
            pred_ans, layer_output_dict = best_model.predict(test_model_input, test_batch_size)
            layer_output_path = data_config.get("layer_output_path", "")
            for key, value in layer_output_dict.items():
                file_name = layer_output_path + f'{model_config.get("model_name", "")}_l2{model_config.get("l2_reg_dnn", "0")}_{key}.pkl'
                with open(file_name, 'wb') as f:
                    pickle.dump(value, f)
            # layer_output_df = pd.DataFrame([layer_output_dict])
            # layer_output_df.to_csv(layer_output_path, index=False, header=True)
        else:
            pred_ans = best_model.predict(test_model_input, test_batch_size)

        test_result_path = data_config.get("test_result_path", "")
        test_result_dict = {}
        model_type = data_config.get("data_name", "") + "_" + model_config.get("task_name", "") + "_" \
               + model_config.get("model_name", "") + "_" + str(seed)
        print(model_type)
        test_result_dict["type"] = model_type
        for i, target_name in enumerate(model.task_types):
            if model.task_name == "msl":
                test_mask_i = torch.Tensor(test_mask[:, i]).bool()
                label = torch.Tensor(np.array(test[target].values)[:, i])
                mask_label = torch.masked_select(label, test_mask_i)
                mask_label = mask_label.cpu().data.numpy().reshape((-1, 1))
                pred = torch.Tensor(np.array(pred_ans[:, i]))
                mask_pred = torch.masked_select(pred, test_mask_i)
                mask_pred = mask_pred.cpu().data.numpy().reshape((-1, 1))
                LogLoss = round(log_loss(mask_label, mask_pred), 4)
                AUC = round(roc_auc_score(mask_label, mask_pred), 4)
                total_auc = roc_auc_score(np.array(test[target].values)[:, 0], np.sum(pred_ans, axis=-1))
                print(f"total AUC {total_auc}")
            elif model.task_name == "mtmsl":
                l = data_config.get("num_domains", 0)
                j = i % l
                test_mask_i = torch.Tensor(test_mask[:, j]).bool()
                label = torch.Tensor(np.array(test[target].values)[:, i])
                mask_label = torch.masked_select(label, test_mask_i)
                mask_label = mask_label.cpu().data.numpy().reshape((-1, 1))
                pred = torch.Tensor(np.array(pred_ans[:, i]))
                mask_pred = torch.masked_select(pred, test_mask_i)
                mask_pred = mask_pred.cpu().data.numpy().reshape((-1, 1))
                LogLoss = round(log_loss(mask_label, mask_pred), 4)
                AUC = round(roc_auc_score(mask_label, mask_pred), 4)
                y_true = test[target].values[:, [0, l]]
                y_pred = np.concatenate((np.sum(pred_ans[:, :l], axis=-1).reshape(len(y_true), 1),
                                         np.sum(pred_ans[:, l:], axis=-1).reshape(len(y_true), 1)), axis=-1)
                total_auc = roc_auc_score(y_true, y_pred)
                print(f"total AUC {total_auc}")
            else:
                LogLoss = round(log_loss(test[target[i]].values, pred_ans[:, i]), 4)
                AUC = round(roc_auc_score(test[target[i]].values, pred_ans[:, i]), 4)
            print("%s test LogLoss" % target_name, LogLoss)
            print("%s test AUC" % target_name, AUC)
            test_result_dict[f"log_loss_{i}"] = LogLoss
            test_result_dict[f"auc_{i}"] = AUC
        if model.task_name in ["msl", "mtmsl"]:
            test_result_dict["total_auc"] = round(total_auc, 4)
        print(test_result_dict)
        test_result_df = pd.DataFrame([test_result_dict])
        if os.path.exists(test_result_path) is False:
            test_result_df.to_csv(test_result_path, index=False, header=True)
        else:
            test_result_df.to_csv(test_result_path, mode='a', index=False, header=False)
        del model
        del best_model




