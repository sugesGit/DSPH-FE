#!/usr/bin/python

# -*- coding: utf-8 -*-
'''
introduce package
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn import metrics
import random, heapq
from sklearn.metrics import roc_curve, auc

'''
dataset
'''
INDEX = [71,234,278,413,557,603,639,649,653,701,720,759,998,1008,1124,1162,1313,1524,1874,1902,2067,2156,2187,2214,2219,2241,2294,
2389,2857,2902,3010,3014,3091,3118,3122,3149,3186,3189,3190,3194,3253,3437,3586,3614,4067,4158,4418,4459,4463,4481,4522,4526,
4639,4864,4868,4918,5134,5188,5219,5345,5584,5607,5656,5660,5764,6056,6074,6286,6376,6425,6439,6596,6606,6659,6736,6772,6795,6893,6976]
INDEX = np.array(INDEX)
gindex = [0,1,2,3,4,5,6,23,28,52]

FFINDEX = [0,1,2,3,4,5,6,22,23,31,48,52,56,63,65,66,68,69,74,75,77,79,84,85,88,90,91,100,101,118,122,125,131]
# FFINDEX = NoneS
def dataset(path, dataindex = 63, labelindex = 64, startdata = 0, flag= False, savepath = ''):
    data = np.loadtxt(open(path,'rb'),delimiter=",",skiprows=0)
    print(data.shape)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    label = data[:, labelindex]
    print(label)
    if FFINDEX is not None:
        data = data[:,FFINDEX]
    elif not flag:
        # top20tf = data[:,startdata:dataindex]
        # top10gf = data[:,gindex]
        # data = np.hstack((top10gf, top20tf))
        data = data[:,startdata:dataindex]  
    else:
        texturedata = data[:,INDEX]
        geometricdata = data[:, 6912:6975]
        data = np.hstack((geometricdata, texturedata))  
        # savepath = './result/top20texture.txt'
        np.savetxt(savepath, data, delimiter = ',')
        data = data[:,:-1] 
        print('INDEX.shape',INDEX.shape, 'data.shape:',data.shape, texturedata.shape, geometricdata.shape, 'label.shape', label.shape)
    return data, label


def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) 
    roc_auc = auc(fpr,tpr) 
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.savefig("./figure/rocTEST1.png")
    plt.show()
    

def Accuracy(data, label, gbdt):
    y_pred = gbdt.predict(data)
    y_predprob = gbdt.predict_proba(data)[:,1]
    acu_curve(label, y_predprob)

    accuracy = metrics.accuracy_score(label, y_pred)
    AUC = metrics.roc_auc_score(label, y_predprob)
    print ("Accuracy : %.4g" % accuracy)
    print ("AUC Score (Train): %f" % AUC)
    return accuracy, AUC
 
def run(trainpath = './data/Original_DATA/geometric_features/jihe_train.csv', testpath = './data/Original_DATA/geometric_features/jihe_test.csv', dataindex = 63, labelindex = 64, startdata = 0, flag = False):
    train_data, train_label = dataset(trainpath, dataindex, labelindex, startdata, flag, './data/TOP20_TextureFeatures/traintop20texture.txt')
    print('train_data.shape',train_data.shape)
    test_data, test_label = dataset(testpath, dataindex, labelindex, startdata, flag, './data/TOP20_TextureFeatures/testtop20texture.txt')
    print('test_data.shape',test_data.shape)
    '''
    preparing the classifier and training it!
    '''
    gbdt = GradientBoostingClassifier(
        init=None,
        learning_rate=0.1,
        loss='deviance',
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        random_state=None,
        subsample=1.0,
        verbose=0,
        warm_start=False)


    print ("fit start!")
    gbdt.fit(train_data[:], train_label[:])
    print ("fit success!")

    '''
    print the result
    '''

    train_accuracy, train_AUC = Accuracy(train_data, train_label, gbdt)
    test_accuracy, test_AUC = Accuracy(test_data, test_label, gbdt)

    score = gbdt.feature_importances_
    top10 = list(map(list(score).index, heapq.nlargest(33, score)))
    print ('gbdt.feature_importances_', gbdt.feature_importances_)
    print (top10)

    result = open("./result/importance_analysis.txt", "a")
    result.write('Accuracy\n')
    result.write('trainset:'+str(train_accuracy)+'     AUC score:'+str(train_AUC)+'\n')
    result.write('testset:'+str(test_accuracy)+'     AUC score:'+str(test_AUC)+"\n")
    result.write('top 10 important feature:'+str(top10)+'\n')
    result.write('gbdt.feature_importances_'+str(gbdt.feature_importances_[top10])+'\n')
    result.flush()
    result.close( )
# trainpath = './data/Original_DATA/geometric_features/jihe_train.csv'
# testpath = './data/Original_DATA/geometric_features/jihe_test.csv'

def main():
    trainpath = './data/TOP20_TextureFeatures/traintop20texture.txt'
    testpath = './data/TOP20_TextureFeatures/testtop20texture.txt'
    startdata = 0
    dataindex = -1
    labelindex = -1
    flag = False
    run(trainpath, testpath, dataindex, labelindex, startdata, flag)

main()
