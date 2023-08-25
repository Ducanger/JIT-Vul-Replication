from sklearn.linear_model import LogisticRegression
import os, time
import pandas as pd
from baselines.utils.performance_measure import PerformanceMeasure
from baselines.utils.results_writer import ResultWriter
from sklearn import preprocessing
import argparse
from sklearn.metrics import accuracy_score

feature_name = ["la", "ld"]
label_name = ["is_buggy_commit"]

def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df

def convert_dtype_dataframe(df, feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    return df

def load_data(base_path: str, baseline_name: str):
    pkl_test = pd.read_pickle(os.path.join(base_path, baseline_name, "features_test.pkl"))
    pkl_train = pd.read_pickle(os.path.join(base_path, baseline_name, "features_train.pkl"))
    
    pkl_train = convert_dtype_dataframe(pkl_train, feature_name)
    pkl_test = convert_dtype_dataframe(pkl_test, feature_name)
    pkl_train = replace_value_dataframe(pkl_train)
    pkl_test = replace_value_dataframe(pkl_test)
    # pkl_train['loc'] = pkl_train['la']+ pkl_train['ld']
    # pkl_test['loc'] = pkl_test['la']+ pkl_test['ld']

    X_train, y_train = pkl_train[['la']].values, pkl_train[label_name].values.flatten()
    X_test, y_test = pkl_test[['la']].values, pkl_test[label_name].values.flatten()

    return X_train, y_train, X_test, y_test

def load_test_dataframe(base_path: str, baseline_name: str):
    pkl_test = pd.read_pickle(os.path.join(base_path, baseline_name, "features_test.pkl"))
    pkl_test = convert_dtype_dataframe(pkl_test, feature_name)
    if 'jitline' in baseline_name:
        pkl_test = pkl_test.sort_values(by='commit_hash')
    # effort
    result_df = pd.DataFrame()
    result_df['commit_id'] = pkl_test['commit_hash']
    result_df['LOC'] = pkl_test['la'] + pkl_test['ld']
    result_df['label'] = list(pkl_test['is_buggy_commit'].values)

    return result_df

def LA_train_and_eval(baseline_name: str = None):
    X_train, y_train, X_test, y_test = load_data(base_path, baseline_name)
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    print(X_train)

    print(f"building model {baseline_name}")
    model = LogisticRegression(max_iter=7000).fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    result_df = load_test_dataframe(base_path, baseline_name)
    result_df['defective_commit_pred'] = y_pred
    result_df['defective_commit_prob'] = y_pred_prob

    presults =  PerformanceMeasure().eval_metrics(result_df=result_df)
    print(presults)
    result_df.to_csv(f'data/{baseline_name}/la_predict.csv', index=False, columns=['commit_id', 'label', 'defective_commit_pred', 'defective_commit_prob'])
    # ResultWriter().write_result(result_path=result_path, method_name="LApredict", presults=presults)

if __name__ == "__main__":
    start_time = time.time()
    arg = argparse.ArgumentParser()
    arg.add_argument('-split', type=str)
    args = arg.parse_args()
    split_data = args.split
    print("Running LA model")
    base_path = "data/"
    # result_path = os.path.dirname(os.path.dirname(__file__)) + '/results/'
    LA_train_and_eval(f'la/{split_data}')
    print("--- %s seconds ---" % (time.time() - start_time))
    
