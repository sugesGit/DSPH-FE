import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm  # SVM算法
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def cross_Validation(clf,data,label):
    scores = cross_val_score(clf, data, label, cv=5)  #cv为迭代次数。
    print(scores)  # 打印输出每次迭代的度量值（准确度）
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#load 数据
print(1)
data = np.loadtxt(open('./texture_features/pca_texture2.csv','rb'),delimiter=" ",skiprows=0)
print(2)
label = np.loadtxt(open("./texture_features/pca_lable_texture2.csv","rb"),delimiter=",",skiprows=0)
print('texture features')

#交叉验证
method= [RandomForestClassifier(),tree.DecisionTreeClassifier(criterion='gini'),GaussianNB(),svm.SVC(kernel='linear', C=1)]
for clf in method:
    print(clf)
    cross_Validation(clf,data,label)



