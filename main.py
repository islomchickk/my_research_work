import os
import pandas as pd
import numpy as np

from my_system.linear_regression import LinearRegression
from my_system.bayessian_lr import BayesianLR
from my_system.catboost_regression import CatboostRegressor
from my_system.lgbm_classifair import PredictModel
from feature_selection_and_catboost import FeatureSelection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, StandardScaler

from pymfe.mfe import MFE


class MinMaxScaler:
    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)

    def transform(self, data):
        return (data-self.min) / (self.max-self.min)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0).dropna()
    df = pd.DataFrame(data=df, columns=df.columns)
    print(df.shape)
    df.columns = df.columns.str.replace('-', '_').str.replace('.', '_').str.replace('(', '_').str.replace(')', '_').str.replace('/', '_')
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = df.loc[:30, :], df.loc[30:, :]
    # train, test = train_test_split(df.astype(float), test_size=0.3, random_state=42, shuffle=True)
    scaler = MinMaxScaler()
    train.iloc[:, -1] = scaler.fit_transform(train.iloc[:, -1].values)
    test.iloc[:, -1] = scaler.transform(test.iloc[:, -1].values)
    return df, train, test

def get_meta(df: pd.DataFrame, mfe: MFE =None):
    if mfe is None:
        mfe = MFE(groups=["general", "statistical", "model-based"])
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    mfe.fit(X, y)
    ft = mfe.extract()
    return pd.DataFrame(data=[ft[1]], columns=ft[0])


def extract_meta_features(root:str, files:list[str], save_path:str =None, path_to_rmse:str =None):
    meta_features = pd.DataFrame()
    mfe = MFE(groups=["general", "statistical", "model-based"])
    scaler = StandardScaler()
    for file in files:
        try:
            print(file)
            file_path = os.path.join(root, file)
            df1 = pd.read_csv(file_path, index_col=0).dropna().select_dtypes(np.number)
            df = pd.DataFrame(data=df1, columns=df1.columns)
            if df.shape[0]>1000:
                df = df.sample(1000, random_state=42)
            temp = get_meta(df, mfe)
            temp['dataset_id'] = file[:-4]
            meta_features = pd.concat([meta_features, temp])
        except:
            pass

    drop_cols = list(meta_features.columns[meta_features.isna().sum() > 10]) + ['lh_trace', 'roy_root']
    meta_features.drop(drop_cols, axis=1, inplace=True)
    meta_features.fillna(meta_features.mean(), inplace=True)
    meta_features.to_csv('sres.csv', index=False)
    meta_features.reset_index(drop=True, inplace=True)
    # meta_features = pd.read_csv("meta_features.csv")
    if path_to_rmse:
        for i, id_ in enumerate(meta_features['dataset_id'].values):
            try:
                df = pd.read_csv(f'{path_to_rmse}/{id_}.csv')
                meta_features.loc[i, 'len'] = df['size'].values[-1]
                print(i, np.argmin(df.loc[len(df)-1, ['bn', 'cat', 'bmb']].values))
                meta_features.loc[i, 'best_model'] = np.argmin(df.loc[len(df)-1, ['bn', 'cat', 'bmb']].values)
            except:
                meta_features.loc[i, 'best_model'] = -1
                print(id_, "don't exsist")
    meta_features = meta_features[meta_features['best_model'] != -1]
    if save_path:
        meta_features.to_csv(save_path, index=False)
    return meta_features

class Predict:
    def __init__(self, ) -> None:
        self.lr = LinearRegression()
        self.cat = CatboostRegressor()
        self.blr = BayesianLR()
        self.pm = PredictModel()
    
    def fit_models(self, meta_features_path):
        self.meta_features = pd.read_csv(meta_features_path)
        self.pm.fit_classification_models(self.meta_features.drop('dataset_id', axis=1))
    
    def predict_target(self, train, X_test):
        train_meta = get_meta(train)        
        train_meta['len'] = len(train)
        train_meta = train_meta.loc[:, list(self.meta_features.drop(['dataset_id', 'best_model'],axis=1).columns)]
        train_meta.fillna(self.meta_features.mean(), inplace=True)
        model_num = self.pm.predict_best_model(train_meta)
        print(model_num)
        result.loc[len(result)+1, 'model_num'] = model_num
        if model_num == 0:
            predict = self.lr.fit_predict(train, X_test)
        elif model_num == 1:
            predict = self.cat.fit_predict(train, X_test)
        else:
            predict = self.blr.fit_predict(train, X_test)
        return predict



if __name__ == "__main__":
    root = 'openml/'
    meta_features_path = 'res.csv'
    path_to_rmse = "results_step10_all_model"
    files = os.listdir(root)[:-1]
    res = extract_meta_features(root, files, meta_features_path, path_to_rmse)
    fs = FeatureSelection()
    predict = Predict()
    predict.fit_models(meta_features_path)
    
    files = os.listdir("openml/")
    idx = np.random.randint(0, len(files), 20)
    save_path = 'rs_vs_fs.csv'
    if os.path.exists(save_path):
        result = pd.read_csv(save_path)
    else:
        result = pd.DataFrame(columns=['file', 'model_num', 'rmse_recsys', 'rmse_feature_extraction'])
    for i in result['file'].values:
        file = i
        print(file)
        if pd.read_csv(f"openml/{file}").shape[0] > 30:
            df, train, test = load_data(f"openml/{file}")
        else:
            continue
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
        print(train.shape, test.shape)
        print('load!')
        recsys_pred = predict.predict_target(train, X_test)
        fs_pred = fs.fit_predict(train, X_test)
        print(y_test.shape, recsys_pred.shape, fs_pred.shape)
        if len(np.where(np.isnan(recsys_pred))[0]):
            nan_indices = np.where(np.isnan(recsys_pred))[0]
            recsys_pred = np.delete(recsys_pred, nan_indices)
            fs_pred = np.delete(fs_pred, nan_indices)
            y_test = np.delete(y_test.values, nan_indices)
            print(y_test.shape, recsys_pred.shape, fs_pred.shape)
        rmse_rs = mean_squared_error(y_test, recsys_pred, squared=False)
        rmse_fs = mean_squared_error(y_test, fs_pred, squared=False)
        print("recsys", rmse_rs)
        print("feature selection", rmse_fs)
        result.loc[len(result), ['file','rmse_recsys', 'rmse_feature_extraction']] = file, rmse_rs, rmse_fs
        result.to_csv('v2_'+save_path, index=False)

