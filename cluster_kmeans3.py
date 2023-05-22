# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# 读取数据文件
train_data = pd.read_csv('MachineLearning\wdbc_train.data', header=None)
test_data = pd.read_csv('MachineLearning\wdbc_test.data', header=None)
validation_data = pd.read_csv('MachineLearning\wdbc_validation.data', header=None)

# 提取特征和标签
X_train = train_data.iloc[:, 2:].values # 去掉ID和诊断结果列
y_train = train_data.iloc[:, 1].values # 只保留诊断结果列
X_test = test_data.iloc[:, 2:].values
y_test = test_data.iloc[:, 1].values
X_validation = validation_data.iloc[:, 2:].values
y_validation = validation_data.iloc[:, 1].values

# 将标签转换为数字，M为1，B为0
y_train = np.where(y_train == 'M', 1, 0)
y_test = np.where(y_test == 'M', 1, 0)
y_validation = np.where(y_validation == 'M', 1, 0)

# 创建KMeans对象，指定聚类数为2，随机种子为0
kmeans = KMeans(n_clusters=2, random_state=0)

# 使用训练数据拟合模型
kmeans.fit(X_train)

# 使用测试数据预测标签
y_pred = kmeans.predict(X_test)

# 计算NMI评估指标
nmi = normalized_mutual_info_score(y_test, y_pred)

# 打印结果
print('The NMI score on the test data is:', nmi)
