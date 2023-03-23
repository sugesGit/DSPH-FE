from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd

def read(path):
    data = pd.read_csv(path, header = None)
    data =  np.array(data)
    random.shuffle(data)
    random.shuffle(data)
    return data
def load_data(path_train, path_test):
    # data = np.loadtxt(open("C://Users/Administrator/Desktop/夏令营/experiment/data/difference/learning.data","rb"),delimiter=",",skiprows=0)

    datatrain = read(path_train)
    datatest = read(path_test)

    data = np.vstack((datatrain, datatest))
    print(data.shape)
    X= data[:, :6912]
    y= data[:, 6976]
    return X,y
           
if __name__=='__main__':
    path_train = "./data/Original_DATA/trainset.txt"
    path_test = "./data/Original_DATA/testset.txt"
    X,y=load_data(path_train, path_test) # 产生用于降维的数据集
    print(X,y)

    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    print(X_scaler)
    x = X_scaler.fit_transform(X)
    print(x.shape)

    # PCA
    pca = PCA(n_components=0.95)# 保证降维后的数据保持95%的信息
    pca
    pca.fit(x)
    result=pca.transform(x)
    # np.savetxt("./data/PCA_DATA/pca_trainset.csv",result[:22613,:])
    # np.savetxt("./data/PCA_DATA/pca_label_trainset.csv",y[:22613])
    # np.savetxt("./data/PCA_DATA/pca_testset.csv",result[22613:,:])
    # np.savetxt("./data/PCA_DATA/pca_label_testset.csv",y[22613:])
    print(result.shape)


#a = [[1, 2],[3, 4],[5,6]];
#random.shuffle(a)
#print(a)
