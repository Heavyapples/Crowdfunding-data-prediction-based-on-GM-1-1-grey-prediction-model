# encoding=utf8 

import numpy as np
import pandas as pd

# 如需添加其他库，请自行添加

STU_NO = 200001111


class MyScaler:
    def __init__(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x_standard):
        return x_standard * self.std + self.mean

# GM(1,1)灰色预测
class Model_GM_11():
    def __init__(self, data):
        assert isinstance(data, np.ndarray), 'data type must be numpy.ndarray'
        self.x_o = data
        self._lenth = len(self.x_o)
        self.x1 = np.zeros(self._lenth)
        self.x1[0] = self.x_o[0]
        self.z1 = np.zeros_like(self.x1)
        for i in range(1, self._lenth):
            self.x1[i] = self.x1[i - 1] + self.x_o[i]
            self.z1[i] = (self.x1[i - 1] + self.x1[i]) / 2
        self.k = None
        self.b = None

    def fit(self):
        x = self.z1[1:]
        y = self.x_o[1:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        self.k = (x - x_mean).dot(y - y_mean) / ((x - x_mean).dot(x - x_mean) + 1e-10)
        self.b = y_mean - self.k * x_mean

    def predict(self):
        assert self.k is not None and self.b is not None, "must run fit before predict"
        pred = (1 - np.exp(self.k)) * (self.x_o[0] - self.b / (self.k + 1e-10)) * np.exp(-self.k * self._lenth)
        return pred

    def assess(self):
        r = list()
        e = list()
        for i in range(1, self._lenth):
            r.append(self.x_o[i] / self.x_o[i - 1])
            e.append(np.abs(1 - (1 - 0.5 * self.k) / (1 + 0.5 * self.k) / r[i - 1]))
        return np.mean(e)


def data_preprocess(df):
    # 处理字符串列
    for column in ['proj_name', 'proj_time']:
        df[column] = df[column].str.strip()

    # 处理数据格式混乱或不正确的问题
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()  # 获取数字列
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # 处理缺失值
    for column in numeric_columns:
        if df[column].isnull().any() or (df[column] == -99).any():
            df[column] = df[column].replace(-99, np.nan)
            median = df[column].median()
            df[column].fillna(median, inplace=True)
    return df


df = pd.read_csv('ZcMidterm.csv', encoding='gbk')

df_preprocessed = data_preprocess(df)

df_preprocessed.to_csv('{}_preprocessed.csv'.format(STU_NO))

data = np.array(df_preprocessed.iloc[:, 2:])

predict = list()
eta_mean = list()
for i in range(data.shape[0]):
    scaler = MyScaler(data[i])
    data_std = scaler.transform(data[i])
    GM = Model_GM_11(data_std)
    GM.fit()
    eta_mean.append(GM.assess())
    pred = GM.predict()
    predict.append(scaler.inverse_transform(pred))

validate = sum(np.array(eta_mean) <= 0.2) / len(eta_mean)
print('validate: ', validate)
predict = np.array(predict).reshape((-1, 1))
project = np.array(df_preprocessed['proj_name']).reshape((-1, 1))
result = pd.DataFrame(np.hstack([project, predict]))
result.columns = ['proj_name', 'predict']
result.to_csv('{}_result.csv'.format(STU_NO), encoding='gbk', index=False)
