# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:21:51 2019

@author: Asus
"""


# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:11:10 2018

@author: asus
"""

#Import Library
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import random
import time
import pandas as pd
#Assumed you have, X (predictor) and Y (target) fortraining data set and x_test(predictor) of test_dataset
print(1)
'''
#texture features
data_train = pd.read_csv("./dataset/pca_trainset.csv",header=None, dtype = float ,sep=' ')
data_test = pd.read_csv("./dataset/pca_testset.csv",header=None, dtype = float,sep= ' ')
label_train = pd.read_csv("./dataset/pca_label_trainset.csv",header=None,dtype=int)
label_test = pd.read_csv("./dataset/pca_label_testset.csv",header=None,dtype=int)
data_train = np.hstack((data_train, label_train))
data_test = np.hstack((data_test, label_test))
X = data_train[:, :2548]
y = data_train[:,2548]
'''
#geometric features
data_train = pd.read_csv("./geometric_features/data/jihe_train.csv",header=None, dtype = float ,sep=',')
data_test = pd.read_csv("./geometric_features/data/jihe_test.csv",header=None, dtype = float,sep= ',')
data_train = np.array(data_train)
data_test = np.array(data_test)
print(2)
# data_train = np.array(data_train.sample(frac=1.0))
# data_test = np.array(data_test.sample(frac=1.0))
for epoch in range(5):
    random.shuffle(data_train)
    random.shuffle(data_test)

print(3)
X = data_train[:, :63]
y = data_train[:,64]
print(X.shape,y.shape)
#vedio_code_train = data_train[:,63]
x_test=data_test[:, :63]
y_test= data_test[:,64]
print(x_test.shape,y_test.shape)
#vedio_code_test = data_test[:,63]
random_num=random.randint(0,len(y_test))
print(y_test[random_num:random_num+10])
print(4)

category0=0
category1=0
for i in range(0, len(y_test)): 
     if y_test[i]==-1:
         category0+=1
     else:
         category1+=1
            
print(category0,category1)

start = time.time()
model= RandomForestClassifier()
model.fit(X, y)
predicted= model.predict(x_test)
print(predicted[random_num:random_num+10])
res=0;
acc=0;
pre_category0=0
pre_category1=0
pre0=0
pre1=0
correct_vedio_code=[]
wrong_vedio_code=[]
for i in range(0, len(y_test)): 
     if predicted[i] == y_test[i]:
        # correct_vedio_code.append(vedio_code_test[i])
         if predicted[i]==-1:
             pre_category0+=1
         else:
             pre_category1+=1
     else:
         #wrong_vedio_code.append(vedio_code_test[i])
         if predicted[i]==-1:
             pre0+=1
         else:
             pre1+=1
acc0=(pre_category0+pre_category1)/(pre0+pre_category0+pre1+pre_category1)
p0=(pre_category0)/(pre_category0+pre0)
call0=pre_category0/(pre_category0+pre1)
F_0=2*p0*call0/(p0+call0)
end = time.time()
print("RandomForeset Accuracy="+str(acc0))
print("RandomForeset Precision="+str(p0))
print("RandomForeset Recall="+str(call0))
print("RandomForeset F1score="+str(F_0))
print('time='+str(end-start))
#np.savetxt("statistic vedio/RandomForesetcorrect_vedio_code2.csv",correct_vedio_code)
#np.savetxt("statistic vedio/RandomForesetwrong_vedio_code2.csv",wrong_vedio_code)

# print(5)
# start = time.time()
# model = tree.DecisionTreeClassifier(criterion='gini')
# model.fit(X, y)
# model.score(X, y)
# predicted= model.predict(x_test)
# print(predicted[random_num:random_num+10])
# res=0;
# acc=0;
# pre_category0=0
# pre_category1=0
# pre0=0
# pre1=0
# correct_vedio_code=[]
# wrong_vedio_code=[]
# for i in range(0, len(y_test)): 
#      if predicted[i] == y_test[i]:
#          #correct_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre_category0+=1
#          else:
#              pre_category1+=1
#      else:
#         # wrong_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre0+=1
#          else:
#              pre1+=1
# acc0=(pre_category0+pre_category1)/(pre0+pre_category0+pre1+pre_category1)
# p0=(pre_category0)/(pre_category0+pre0)
# call0=pre_category0/(pre_category0+pre1)
# F_0=2*p0*call0/(p0+call0)
# end = time.time()
# print("DecisionTree Accuracy="+str(acc0))
# print("DecisionTree Precision="+str(p0))
# print("DecisionTree Recall="+str(call0))
# print("DecisionTree F1score="+str(F_0))
# print('time='+str(end-start))
# #np.savetxt("statistic vedio/DecisionTreecorrect_vedio_code2.csv",correct_vedio_code)
# #np.savetxt("statistic vedio/DecisionTreewrong_vedio_code2.csv",wrong_vedio_code)

# print(6)
# start = time.time()
# model = GaussianNB() 
# model.fit(X, y)
# predicted= model.predict(x_test)
# print(predicted[random_num:random_num+10])
# res=0;
# acc=0;
# pre_category0=0
# pre_category1=0
# pre0=0
# pre1=0
# correct_vedio_code=[]
# wrong_vedio_code=[]
# for i in range(0, len(y_test)): 
#      if predicted[i] == y_test[i]:
#          #correct_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre_category0+=1
#          else:
#              pre_category1+=1
#      else:
#          #wrong_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre0+=1
#          else:
#              pre1+=1
# acc0=(pre_category0+pre_category1)/(pre0+pre_category0+pre1+pre_category1)
# p0=(pre_category0)/(pre_category0+pre0)
# call0=pre_category0/(pre_category0+pre1)
# F_0=2*p0*call0/(p0+call0)
# end = time.time()
# print("Bayesian Accuracy="+str(acc0))
# print("Bayesian Precision="+str(p0))
# print("Bayesian Recall="+str(call0))
# print("Bayesian F1score="+str(F_0))
# print('time='+str(end-start))
# #np.savetxt("statistic vedio/Bayesiancorrect_vedio_code2.csv",correct_vedio_code)
# #np.savetxt("statistic vedio/Bayesianwrong_vedio_code2.csv",wrong_vedio_code)

# print(7)
# start = time.time()
# model = svm.SVC() 
# model.fit(X, y)
# model.score(X, y)
# predicted= model.predict(x_test)
# print(predicted[random_num:random_num+10])
# res=0;
# acc=0;
# pre_category0=0
# pre_category1=0
# pre0=0
# pre1=0
# correct_vedio_code=[]
# wrong_vedio_code=[]
# for i in range(0, len(y_test)): 
#      if predicted[i] == y_test[i]:
#          #correct_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre_category0+=1
#          else:
#              pre_category1+=1
#      else:
#          #wrong_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre0+=1
#          else:
#              pre1+=1
# acc0=(pre_category0+pre_category1)/(pre0+pre_category0+pre1+pre_category1)
# p0=(pre_category0)/(pre_category0+pre0)
# call0=pre_category0/(pre_category0+pre1)
# F_0=2*p0*call0/(p0+call0)
# end = time.time()
# print("SVM Accuracy="+str(acc0))
# print("SVM Precision="+str(p0))
# print("SVM Recall="+str(call0))
# print("SVM F1score="+str(F_0))
# print('time='+str(end-start))
# #np.savetxt("statistic vedio/SVMcorrect_vedio_code2.csv",correct_vedio_code)
# #np.savetxt("statistic vedio/SVMwrong_vedio_code2.csv",wrong_vedio_code)

# print(8)
# start = time.time()
# model=KNeighborsClassifier(n_neighbors=6)
# model.fit(X, y)
# predicted= model.predict(x_test)
# print(predicted[random_num:random_num+10])
# res=0;
# acc=0;
# pre_category0=0
# pre_category1=0
# pre0=0
# pre1=0
# correct_vedio_code=[]
# wrong_vedio_code=[]
# for i in range(0, len(y_test)): 
#      if predicted[i] == y_test[i]:
#          #correct_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre_category0+=1
#          else:
#              pre_category1+=1
#      else:
#          #wrong_vedio_code.append(vedio_code_test[i])
#          if predicted[i]==-1:
#              pre0+=1
#          else:
#              pre1+=1
# acc0=(pre_category0+pre_category1)/(pre0+pre_category0+pre1+pre_category1)
# p0=(pre_category0)/(pre_category0+pre0)
# call0=pre_category0/(pre_category0+pre1)
# F_0=2*p0*call0/(p0+call0)
# end = time.time()
# print("KNN Accuracy="+str(acc0))
# print("KNN Precision="+str(p0))
# print("KNN Recall="+str(call0))
# print("KNN F1score="+str(F_0))
# print('time='+str(end-start))
# #np.savetxt("statistic vedio/KNNcorrect_vedio_code2.csv",correct_vedio_code)
# #np.savetxt("statistic vedio/KNNwrong_vedio_code2.csv",wrong_vedio_code)

