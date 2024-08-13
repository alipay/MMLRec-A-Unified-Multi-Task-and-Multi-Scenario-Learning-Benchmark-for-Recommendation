import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from model.utils import SparseFeat, DenseFeat, get_feature_names
from tqdm import tqdm
import json
import os

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

def ctrdataset(config):
    data_config = config['data_config']
    model_config = config['model_config']
    train_path = data_config.get("train_dataset_path", "")
    test_path = data_config.get("test_dataset_path", "")
    all_columns = data_config.get("all_columns", [])
    feature_columns = data_config.get("feature_columns", [])
    dense_columns = data_config.get("dense_columns", [])
    ignore_columns = data_config.get("ignore_columns", [])
    label_columns = data_config.get("label_columns", ['label'])
    train_df = pd.read_csv(train_path, usecols=all_columns)
    test_df = pd.read_csv(test_path, usecols=all_columns)

    if 'kuairec' in train_path:
        for col in all_columns:
            if 'onehot' in col:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)
        train_df = train_df[train_df['user_active_degree'] != '0']

    if 'iaac' in train_path:
        train_df['predict_category_property'] = train_df['predict_category_property'].astype(str)
        test_df['predict_category_property'] = test_df['predict_category_property'].astype(str)
        print(len(test_df))
        test_df = test_df[:-2]
        print(len(test_df))

    train_len = len(train_df)
    df = pd.concat([train_df, test_df])

    task_name = model_config.get("task_name", "mtl")
    mask_column = data_config.get('mask_column', '')
    scene_feature = data_config.get("scene_feature", "")
    emb = model_config.get("emb", 4)

    if scene_feature != "" and scene_feature not in feature_columns:
        feature_columns.append(scene_feature)

    sparse_features = feature_columns
    for col in tqdm(all_columns):
        if col not in label_columns + ignore_columns:
            if 'amazon_new' in train_path:
                df[col] = df[col].astype(str)
            if col in dense_columns:
                mms = MinMaxScaler()
                df[col] = mms.fit_transform(df[[col]]).reshape(-1)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    

    new_columns = sparse_features + dense_columns + label_columns
    if task_name in ['msl', 'mtmsl'] and mask_column != '':
        if mask_column not in feature_columns:
            new_columns += [mask_column]

    df = df.reindex(columns=new_columns)
    print(df.head())

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].max() + 1, embedding_dim=emb)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_columns]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train, test = df[:train_len], df[train_len:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    test_mask = None
    if task_name in ['msl', 'mtmsl'] and mask_column != '':
        if mask_column not in feature_columns:
            train_model_input[mask_column] = train[mask_column]
            test_model_input[mask_column] = test[mask_column]
        mask_values = data_config.get('mask_values', [])
        num_domains = data_config.get('num_domains', 1)
        domain_values = test[mask_column]
        test_mask = get_test_mask(domain_values, mask_values, num_domains)
        print(train[mask_column].value_counts())
        print(test[mask_column].value_counts())
    return train, test, test_mask, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns

def get_test_mask(domain_values, mask_values, num_domains):
    domain_values = np.tile(np.array(domain_values).reshape((-1, 1)), (1, num_domains))
    mask_values = np.tile(np.array(mask_values).reshape((1, -1)), (len(domain_values), 1))
    domain_mask = (domain_values == mask_values).astype(np.float32)
    return domain_mask

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "ny":
        return np.load(path)
    elif suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)