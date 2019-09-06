# encoding: utf-8
# 2019/09/06
# in: import.csv
# csv output
#
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import time
import datetime as dt
#from db_mongo_func import dbMongoFunc
#from com_measure_func import ComMeasureFunc

#
def conv_tm2float(nDim ):
	ret=[]
	for item in nDim:
		ret.append(item.total_seconds())
	return ret
###########################
# main
###########################
#rdDim = pd.read_csv("import.csv", names=('date', 'hnum', 'lnum') )
rdDim = pd.read_csv("import.csv" )
rdDim['time_2'] = pd.to_datetime(rdDim['date'])
# Y
Y = rdDim["H"]
Y = np.array(Y, dtype = np.float32).reshape(len(Y) ,1)
# X
xAxis= np.array(rdDim['time_2'].tolist() )

min = rdDim['time_2'].min()
rdDim['diff'] = rdDim['time_2'] -min
diff = conv_tm2float(rdDim['diff'] )
xDim =np.array(diff )

X = np.array(xDim, dtype = np.float32).reshape(len(xDim ) ,1)
#X = np.array(xDim.tolist() )

print ("start...")
start_time = time.time()

# 予測モデルを作成
clf = linear_model.LinearRegression()
clf.fit(X, Y)

# 回帰係数
print(clf.coef_)
# 切片 (誤差)
print(clf.intercept_)
# 決定係数
print(clf.score(X, Y))

#predict
pred = clf.predict(X)

interval = int(time.time() - start_time)
print ("実行時間: {}sec".format(interval) )
#print(pred.shape)
d = pred.tolist()
predArr = []
for item in d:
	predArr.append(item[0] )

arr = {
	'date': xAxis.tolist(),
    'Y': rdDim["H"].tolist(),
	'pred' : predArr
}
df = pd.DataFrame(arr)
df.to_csv('out_pred.csv')
