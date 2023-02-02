import pandas as pd
import numpy as np
import os
import gc
import time
from datetime import datetime
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import warnings

base_seed = 0

DEBUG = False
TEST = True

train = pd.read_pickle("../input/amex-feature-engineering/traindata_agg.pkl")
train_labels = (
    pd.read_csv("../input/amex-default-prediction/train_labels.csv")
    .set_index("customer_ID")
    .loc[train.index]
)

from constant import catagorical_columns, num_cols

# Preprocessing

# Onehot Encoding
train_ = pd.get_dummies(train[catagorical_columns])
train = pd.concat([train, train_], axis=1)
train = train.drop([catagorical_columns], axis=1)

# Normalize
train = scaler = StandardScaler()
train = scaler.transform(num_cols)

# To calculate mean use imputer class
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(train)
X = imputer.transform(train)


def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(
        df1, columns=[col + "_diff1" for col in df[num_features].columns]
    )
    df1["customer_ID"] = customer_ids
    return df1


### Drop values
features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
cat_features = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
num_features = [col for col in features if col not in cat_features]
train_num_agg = train.groupby("customer_ID")[num_features].agg(
    ["first", "mean", "std", "min", "max", "last"]
)
train_num_agg.columns = ["_".join(x) for x in train_num_agg.columns]
train_num_agg.reset_index(inplace=True)

# Lag Features
for col in train_num_agg:
    for col_2 in ["first", "mean", "std", "min", "max"]:
        if "last" in col and col.replace("last", col_2) in train_num_agg:
            train_num_agg[col + "_lag_sub"] = (
                train_num_agg[col] - train_num_agg[col.replace("last", col_2)]
            )
            train_num_agg[col + "_lag_div"] = (
                train_num_agg[col] / train_num_agg[col.replace("last", col_2)]
            )

train_cat_agg = train.groupby("customer_ID")[cat_features].agg(
    ["count", "first", "last", "nunique"]
)
train_cat_agg.columns = ["_".join(x) for x in train_cat_agg.columns]
train_cat_agg.reset_index(inplace=True)
train_labels = pd.read_csv("../input/amex-default-prediction/train_labels.csv")
# Transform float64 columns to float32
cols = list(train_num_agg.dtypes[train_num_agg.dtypes == "float64"].index)
for col in tqdm(cols):
    train_num_agg[col] = train_num_agg[col].astype(np.float32)
# Transform int64 columns to int32
cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == "int64"].index)
for col in tqdm(cols):
    train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
# Get the difference
train_diff = get_difference(train, num_features)
train = (
    train_num_agg.merge(train_cat_agg, how="inner", on="customer_ID")
    .merge(train_diff, how="inner", on="customer_ID")
    .merge(train_labels, how="inner", on="customer_ID")
)
del train_num_agg, train_cat_agg, train_diff
gc.collect()
test = pd.read_parquet("../input/amex-data-integer-dtypes-parquet-format/test.parquet")
print("Starting test feature engineer...")
test_num_agg = test.groupby("customer_ID")[num_features].agg(
    ["first", "mean", "std", "min", "max", "last"]
)
test_num_agg.columns = ["_".join(x) for x in test_num_agg.columns]
test_num_agg.reset_index(inplace=True)

# Lag Features
for col in test_num_agg:
    for col_2 in ["first", "mean", "std", "min", "max"]:
        if "last" in col and col.replace("last", col_2) in test_num_agg:
            test_num_agg[col + "_lag_sub"] = (
                test_num_agg[col] - test_num_agg[col.replace("last", col_2)]
            )
            test_num_agg[col + "_lag_div"] = (
                test_num_agg[col] / test_num_agg[col.replace("last", col_2)]
            )

test_cat_agg = test.groupby("customer_ID")[cat_features].agg(
    ["count", "first", "last", "nunique"]
)
test_cat_agg.columns = ["_".join(x) for x in test_cat_agg.columns]
test_cat_agg.reset_index(inplace=True)
# Transform float64 columns to float32
cols = list(test_num_agg.dtypes[test_num_agg.dtypes == "float64"].index)
for col in tqdm(cols):
    test_num_agg[col] = test_num_agg[col].astype(np.float32)
# Transform int64 columns to int32
cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == "int64"].index)
for col in tqdm(cols):
    test_cat_agg[col] = test_cat_agg[col].astype(np.int32)
# Get the difference
test_diff = get_difference(test, num_features)
test = test_num_agg.merge(test_cat_agg, how="inner", on="customer_ID").merge(
    test_diff, how="inner", on="customer_ID"
)
del test_num_agg, test_cat_agg, test_diff
gc.collect()

# Add missing values


Features = train.columns

n_fold = 2 if DEBUG else 3
n_seed = 1 if TEST else (2 if DEBUG else 3)
n_estimators = 100 if DEBUG else 10000

kf = StratifiedKFold(n_splits=n_fold)


# Try with SVM, KNN
clf1 = svm.SVC()
clf2 = KNeighborsClassifier(n_neighbors=5)


importances = []
importances_split = []
models = {}
df_scores = []

ids_folds = {}
preds_tr_va = {}

SAMPLE = False


for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train_labels)):

    if TEST:
        if fold > 0:
            continue

    ids_folds[fold] = (idx_tr, idx_va)

    X_tr = train[Features].iloc[idx_tr]
    X_va = train[Features].iloc[idx_va]
    y_tr = train_labels.iloc[idx_tr]
    y_va = train_labels.iloc[idx_va]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va, label=y_va)

    for seed in range(n_seed):
        print("Fold: " + str(fold) + " - seed: " + str(seed))
        key = str(fold) + "-" + str(seed)

        parameters = {
            "objective": "cross_entropy_lambda",
            "boosting": "dart",
            "learning_rate": 0.005,
            #'min_child_samples': 1000,
            "reg_alpha": 10,
            "feature_fraction": 0.3,
            "bagging_fraction": 0.3,
            "max_depth": 6,
            "seed": seed,
            "n_estimators": n_estimators,
            "verbose": -1,
            "linear_tree": True,
        }

        clf = lgb.train(
            parameters,
            lgb_train,
            valid_sets=[lgb_train, lgb_val_data],
            verbose_eval=100,
            feval=amex_metric_mod_lgbm,
            early_stopping_rounds=200,
        )

        preds_tr = pd.Series(clf.predict(X_tr)).rename("prediction")
        preds_va = pd.Series(clf.predict(X_va)).rename("prediction")

        preds_tr_va[(fold, seed)] = (preds_tr, preds_va)

        score = amex_metric(y_va.reset_index(drop=True), preds_va)
        models[key] = clf
        df_scores.append((fold, seed, score))
        print(f"Fold: {fold} - seed: {seed} - score {score:.2%}")
        importances.append(clf.feature_importance(importance_type="gain"))
        importances_split.append(clf.feature_importance(importance_type="split"))


df_sub = pd.read_csv("../input/amex-default-prediction/sample_submission.csv")
df_sub.prediction = proba_series.loc[df_sub.customer_ID].values

df_sub = df_sub.set_index("customer_ID")
df_sub.to_csv("submission.csv")
