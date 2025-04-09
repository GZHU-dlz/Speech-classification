import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import StratifiedKFold

train_data = pd.read_csv("/private/Coswara-Data/cut_5s_data/heathy_data/path_csv/healthy_quality12_5s_vowel_a.csv")
# 拆分数据
print("拆分数据中。。。")
split_ratio = 0.85  # 75%的训练数据
seed = 5  # 随机种子
 
# 分割训练集与测试集
xtrain, xtest, ytrain, ytest = train_test_split(
    train_data, range(train_data.shape[0]), test_size=split_ratio, random_state=seed)
 
print(xtest)
print("------------------")
print(xtrain)
print("------------------")
print(ytrain)
print("------------------")
print(ytest)
