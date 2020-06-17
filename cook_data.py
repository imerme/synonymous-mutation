# amira
# 20200616
# rainy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

# 处理训练集：
train_path = r'./data/raw data/39_NearMiss2.txt'
data = pd.read_csv(train_path, header=0, sep=',', index_col=0)
y_data, x_data = data.loc[:, 'class'], data.iloc(axis=1)[1:]  # 读取标签和特征
MMS = MinMaxScaler()  # 归一化器
SI = SimpleImputer(strategy='constant', fill_value=0)  # 填充器
pipeline = Pipeline([('SI', SI), ('MMS', MMS)])  # 数据处理管道
x_train = pipeline.fit_transform(x_data)  # 对训练数据进行归一化
y_train = np.eye(2)[y_data.astype(int)]  # 特征向量独热编码
np.save(r'./data/cooked data/x_train', x_train)
np.save(r'./data/cooked data/y_train', y_train)  # 保持处理过的数据

# 处理测试集96+96
neg_test = pd.read_csv(r'./data/raw data/test_neg.vcf', header=None, na_values=['na', 'nan'], sep='\t')
pos_test = pd.read_csv(r'./data/raw data/test_pos.vcf', header=None, na_values=['na', 'nan'], sep='\t')
x_test = pd.concat([pos_test, neg_test], axis=0).loc(axis=1)[6:]
x_test = pipeline.transform(x_test)
y_test = np.r_[np.eye(2)[np.ones((96,), dtype=int)], np.eye(2)[np.zeros((96,), dtype=int)]]
np.save(r'./data/cooked data/x_test', x_test)
np.save(r'./data/cooked data/y_test', y_test)


# 处理独立测试集1， 2
# Independent test 1， 246 + 246
pos_test_1 = r'./data/raw data/20200507-三套新独立测试集/HGMD_VariSNP/40feature/pos_test_1.csv'
pos_test_1 = pd.read_csv(pos_test_1, header=0, sep='\t', usecols=list(range(6, 45)), na_values=['na', 'nan'])
neg_test_1 = r'./data/raw data/20200507-三套新独立测试集/HGMD_VariSNP/40feature/neg_test_1.csv'
neg_test_1 = pd.read_csv(neg_test_1, header=0, sep='\t', usecols=list(range(6, 45)), na_values=['na', 'nan'])
x_test_1 = pd.concat([pos_test_1, neg_test_1], axis=0)
x_test_1 = pipeline.transform(x_test_1)
y_test_1 = np.r_[np.eye(2)[np.ones((246,), dtype=int)], np.eye(2)[np.zeros((246,), dtype=int)]]
np.save(r'./data/cooked data/x_test_1', x_test_1)
np.save(r'./data/cooked data/y_test_1', y_test_1)

# Independent test 2, 93 + 5039
pos_test_2 = r'./data/raw data/20200507-三套新独立测试集/IDSV_Clinvar/40feature/pos_test_2.csv'
pos_test_2 = pd.read_csv(pos_test_2, header=None, sep='\t', usecols=list(range(6, 45)), na_values=['na', 'nan'])
neg_test_2 = r'./data/raw data/20200507-三套新独立测试集/IDSV_Clinvar/40feature/neg_test_2.csv'
neg_test_2 = pd.read_csv(neg_test_2, header=None, sep='\t', usecols=list(range(6, 45)), na_values=['na', 'nan'])
x_test_2 = pd.concat([pos_test_2, neg_test_2], axis=0)
x_test_2 = pipeline.transform(x_test_2)
y_test_2 = np.r_[np.eye(2)[np.ones((93,), dtype=int)], np.eye(2)[np.zeros((5039,), dtype=int)]]
np.save(r'./data/cooked data/x_test_2', x_test_2)
np.save(r'./data/cooked data/y_test_2', y_test_2)
