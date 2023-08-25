import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import time
import pickle
import argparse
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def load_data(split_data='by_time'):
    train = pd.read_pickle(f'data/vccfinder/{split_data}/features_train.pkl')
    test = pd.read_pickle(f'data/vccfinder/{split_data}/features_test.pkl')
    return train, test

def preprocess(train, test):
    target = 'label'
    features = train.columns.drop(['commit_id', target])
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    scaler = QuantileTransformer(n_quantiles=10, output_distribution='uniform')
    for col in X_train.columns:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.fit_transform(X_test[[col]])
    return X_train, y_train, X_test, y_test

def train_and_validate(X_train, y_train, X_test, y_test):
    model = LinearSVC(max_iter=200000, class_weight={0: 1, 1:1})
    model.fit(X_train, y_train)
    y_pred = model._predict_proba_lr(X_test)
    y_pred = np.array([y_pred[i][1] for i in range(len(y_pred))])
    thresh = 0.5
    print(f"Precision:  {precision_score(y_test, y_pred >= thresh):.4f}")
    print(f"Recall:  {recall_score(y_test, y_pred >= thresh):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred >= thresh):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred >= thresh):.4f}")
    print(f"AUC:  {roc_auc_score(y_test, y_pred):.4f}")

def pipeline(split_data='by_time'):
    train, test = load_data(split_data)
    X_train, y_train, X_test, y_test = preprocess(train, test)
    train_and_validate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    start_time = time.time()
    arg = argparse.ArgumentParser()
    arg.add_argument('-split', type=str)
    args = arg.parse_args()
    split_data = args.split
    print("Running VCCFinder")
    pipeline(split_data)
    print("--- %s seconds ---" % (time.time() - start_time))