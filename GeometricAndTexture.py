'''
packages
'''
import pandas as pd                         
import numpy as np
import csv

'''
read files
capture geometric features
'''
g_path = "./data/Original_DATA/geometric_features/jihe_train.csv"
g_data = pd.read_csv(g_path, header = None)
g_data =  np.array(g_data)
participants_sequence = g_data[:,63]   #获取每条特征的参与者编号，记作participant
# g_features = g_data[:,:63]
g_features = g_data
participants_num = np.unique(participants_sequence)
print('number of participants', participants_num.shape)

'''
read files
capture texture features
'''
t_rootpath = './data/Original_DATA/texture_features/'

#participants需手动修改，要与几何特征数据集顺序保持一致
# participants = [516, 775, 605, 504, 723, 
#                 462, 592, 868, 565, 574, 
#                 739, 455, 905, 917, ] #测试集
participants = [531, 884, 488, 651, 470,    #1
                493, 897, 535, 872,
                685, 856, 523, 664, 640,
                746, 893, 473, 715, 751, 
                880, 743, 497, 842, 760,        #5
                768, 888, 764, 451, 582,
                729, 569, 467, 779, 783,
                719, 561, 860, 512, 527,
                909, 445, 846, 508, 447,
                864, 913, 852, 578, 458,        #10
                901, 539, 787, 838, ]
sum_feature = []
savepath = './data/Original_DATA/trainset.txt'
flag = True
for participant in participants:
    participant = int(participant)
    num = participants_sequence.tolist().count(participant) #每个人的几何特征数量
    geometric = g_features[300:num-6,]
    g_features = g_features[num:,:]
    
    #获取纹理特征
    t_path = t_rootpath + '0' + str(participant) + '.csv'
    t_data = pd.read_csv(t_path, header = None)
    texture =  np.array(t_data)

    # print(geometric.shape)
    # print(texture.shape)
    if geometric.shape[0]==texture.shape[0]:
        feature = np.hstack((texture, geometric))
    else:
        print('warning!!!!','participant', participant,'geometric.shape',geometric.shape, 'texture.shape',texture.shape )
        texture = texture[:geometric.shape[0],:]
    
    if flag:
        sum_feature = feature
        # print('sum', sum_feature.shape)
        flag = False
    else:
        sum_feature = np.append(sum_feature, feature, axis=0)
        # print('sum', sum_feature.shape)

    print('participant', participant,'geometric.shape',geometric.shape, 'texture.shape',texture.shape,'sum', sum_feature.shape )

np.savetxt(savepath, sum_feature, delimiter = ',') 

print('finished!')