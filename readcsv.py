import pandas as pd
import numpy as np

def read(path):
    data = pd.read_csv(path, header = None)
    data =  np.array(data)
    return data

path = "./dataset/pca_testset.csv"
data = read(path)
print(data[0].shape)