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
    X,y=load_data(path_train, path_test)
    print(X,y)

    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    print(X_scaler)
    x = X_scaler.fit_transform(X)
    print(x.shape)

    # PCA
    pca = PCA(n_components=0.95)
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
