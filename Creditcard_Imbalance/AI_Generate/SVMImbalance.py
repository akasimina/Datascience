import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载数据
data = pd.read_csv('../creditcard.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# PCA降维
pca = PCA(n_components=10)  # 选择10个主成分
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test)

# 支持向量机训练
svm = SVC(random_state=42)
svm.fit(X_train_pca, y_train_resampled)

# 预测和评估
y_pred = svm.predict(X_test_pca)
print(classification_report(y_test, y_pred))