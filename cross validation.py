import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm  # SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def cross_Validation(clf,data,label):
    scores = cross_val_score(clf, data, label, cv=5) 
    print(scores)  # print accuracy
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#load data
print(1)
data = np.loadtxt(open('./texture_features/pca_texture2.csv','rb'),delimiter=" ",skiprows=0)
print(2)
label = np.loadtxt(open("./texture_features/pca_lable_texture2.csv","rb"),delimiter=",",skiprows=0)
print('texture features')

#cross validation
method= [RandomForestClassifier(),tree.DecisionTreeClassifier(criterion='gini'),GaussianNB(),svm.SVC(kernel='linear', C=1)]
for clf in method:
    print(clf)
    cross_Validation(clf,data,label)
