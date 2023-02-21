import tarfile

# Necessary imports
import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE

#head=0表示从从index为0的行开始读，使用.abs()取正数
df = pd.read_csv(r"result.csv", header=0)

col3 = df.columns[0]

print(col3)

colsize = df.columns.size

#df[:, 0]这种写法只有ndarray支持，所以需要转换一下
values = df.values
# target, data = values[:, (colsize-15)], values[:, 5:colsize]  # 脚板
target, data = values[:, (colsize-16)], values[:, 5:colsize]  # 拱高

#train_test_split这个方法可以很方便地划分测试数据和训练数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.2, random_state=125)


#回归
# xg.XGBRegressor
#model = xg.XGBRFRegressor(learning_rate=0.2)


# Instantiation
model = xg.XGBRegressor(objective='reg:linear',
                        n_estimators=10, seed=123)

print("模型参数：", model)

# Fitting the model
model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xtest)

# RMSE Computation
rmse = np.sqrt(MSE(Ytest, Ypred))
print("RMSE : % f" % (rmse))


for i in range(Ypred.size):
    print("预测值：", Ypred[i], "实际值：", Ytest[i])

fig = plt.figure(figsize=(12, 6))

#MSE(Mean Squared Error) 均方误差是指参数估计值与参数真值之差平方的期望值
#MSE可以评价数据的变化成都，MSE的值越小，说明预测模型描述试验数据具有更好的精确度
MSE = metrics.mean_squared_error(Ytest, Ypred)

#RMSE(Root Mean Squard Error)均方根误差
RMSE = np.sqrt(metrics.mean_squared_error(Ytest, Ypred))

print('MSE:', MSE)
print('RMSE:', RMSE)

# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.sans-serif'] = "SimHei"
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.unicode_minus'] = False

#绘图
# plt.plot(range(Xtest.shape[0]), Xtest, color='blue', linewidth=1.5, linestyle='-')
# plt.plot(range(Xtest.shape[0]), Ypred, color='red', linewidth=1.5, linestyle='-')
# plt.legend(['原始值', '预测值'])

plt.plot(Xtest, Ypred)



plt.show()