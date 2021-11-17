import numpy as np
from sklearn.manifold import TSNE
from scipy.io import loadmat  
import matplotlib.pyplot as plt 

def data_preperation(train_data, test_data):
    ''' Transfer data to binary form. '''
    hashes_db = np.concatenate((test_data, train_data),axis=0)
    hashes_db = hashes_db - np.dot(np.ones([hashes_db.shape[0], 1]), np.mean(hashes_db, axis = 0, keepdims = True) )
    binary = np.zeros(hashes_db.shape, dtype = np.float64)
    binary[hashes_db >= 0] = 1
    return binary[0:train_data.shape[0],:], binary[train_data.shape[0]:,:]

def ALLINONE(rootpath, folderiter):
    test_lable = os.path.join(rootpath,  "sat_4_test_lable.mat")
    train_lable = os.path.join(rootpath, "sat_4_train_lable.mat")
    test_data = os.path.join(rootpath,   "sat_4_test.mat")
    train_data = os.path.join(rootpath,  "sat_4_train.mat")
    
    #one = np.array([0, 1, 2, 3, 4 , 5], dtype= np.int8)
    test_lable = loadmat(test_lable)["sat_4_test_lable"][:1000,:]
    one = np.arange(len(test_lable[0]), dtype= np.int8) 
    test_lable = np.matmul(test_lable, one)
    train_lable = loadmat(train_lable)["sat_4_train_lable"][:1000,:]
    train_lable = np.matmul(train_lable, one)
    test_data = loadmat(test_data)["sat_4_test"][:1000,:]
    train_data = loadmat(train_data)["sat_4_train"][:1000,:] 
 
    lantentz = train_data[:, :]
    labelz = np.reshape(train_lable, (1000,))
    lantent_z_tsn = TSNE(n_components=2).fit_transform(lantentz) 
  
    x = lantent_z_tsn[labelz == 0][:,0]
    y = lantent_z_tsn[labelz == 0][:,1]
    plt.plot(x, y, 'o',color='b', markersize=5) 
    
    x = lantent_z_tsn[labelz == 1][:,0]
    y = lantent_z_tsn[labelz == 1][:,1]
    plt.plot(x, y, 'o',color='g', markersize=5) 
    
    x = lantent_z_tsn[labelz == 2][:,0]
    y = lantent_z_tsn[labelz == 2][:,1]
    plt.plot(x, y, 'o',color='r', markersize=5) 
    
    x = lantent_z_tsn[labelz == 3][:][:,0]
    y = lantent_z_tsn[labelz == 3][:,1]
    plt.plot(x, y, 'o',color='c', markersize=5) 
    
    x = lantent_z_tsn[labelz == 4][:][:,0]
    y = lantent_z_tsn[labelz == 4][:][:,1]
    plt.plot(x, y, 'o',color='m', markersize=5)  

    x = lantent_z_tsn[labelz == 5][:][:,0]
    y = lantent_z_tsn[labelz == 5][:][:,1]
    plt.plot(x, y, 'o',color='y', markersize=5) 

    savepath = os.path.join("./logs", folderiter + 'tsne.png')
    plt.savefig(savepath, bbox_inches='tight')  
    
    plt.cla()
    '''
    x=train_data.flatten()#生成【0-100】之间的100个数据,即 数据集
    bins=np.arange(-5,5,0.5)#设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    #直方图会进行统计各个区间的数值
    plt.hist(x,bins,color='fuchsia',alpha=0)#alpha设置透明度，0为完全透明
    plt.xlabel('scores')
    plt.ylabel('count')
    plt.xlim(-5,5)#设置x轴分布范围
    savepath = os.path.join("./logs", folderiter + 'dstri.png')
    plt.savefig(savepath, bbox_inches='tight')  
    plt.cla()
    '''

if __name__ == '__main__': 
    path = "./logs/test_lantent_z/"
    import os
    folderlist = os.listdir(path)
  
    rangkingscore = {}
    #folderlist = ['SAT-6_VitVAE[1, 1, 0.1]64']
    for folderiter in folderlist: 
        sonfolder = os.path.join(path, folderiter)
        if sonfolder[-5:] == ".ckpt":
            continue
        epochs = os.listdir(sonfolder)
        epochs = ['epoch_4']

        for ep in epochs:
            pr = os.path.join(sonfolder, ep) + "/" 
            if ep[-4:] == ".png":
                continue
            try:
                ALLINONE(pr, folderiter)
            except:
                pass  

    print("end")