# 计算 ap 精度 
import numpy as np 
from scipy.io import loadmat
import os 
def compute_hamming_dist(a, b):
    """
    Computes hamming distance vector wisely
    Args:
        a: row-ordered codes {0,1}
        b: row-ordered codes {0,1}
    Returns:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_m1 = a - 1
    b_m1 = b - 1
    c1 = np.matmul(a, b_m1.T)
    c2 = np.matmul(a_m1, b.T)
    return np.abs(c1 + c2)
def gen_sim_mat(class_list1, class_list2):
    """
    Generates a similarity matrix
    Args:
        class_list1: row-ordered class indicators
        class_list2:
    Returns: [N1 N2]
    """
    c1 = np.asarray(class_list1)
    c2 = np.asarray(class_list2)
    sim_mat = np.matmul(c1, c2.T)
    sim_mat[sim_mat > 0] = 1
    return sim_mat 

def data_preperation(train_data, test_data):
    ''' Transfer data to binary form. '''    
    hashes_db = np.concatenate((train_data, test_data), axis=0)  
    hashes_db = hashes_db - np.dot(np.ones([hashes_db.shape[0], 1]), np.mean(hashes_db, axis = 0, keepdims = True))
    binary = np.zeros(hashes_db.shape, dtype = np.float64)
    binary[hashes_db >= 0] = 1 
    return binary[0:train_data.shape[0],:], binary[train_data.shape[0]:,:]


def get_topk_precise(db_data, query_data, query_label, db_label, topk=10): 
    distances = compute_hamming_dist(query_data, db_data) 
    dist_argsort = np.argsort(distances)
    sim_mat = gen_sim_mat(query_label, db_label)
    precX = np.zeros([query_data.shape[0]])    
    for i in range(query_data.shape[0]): 
        serchimgdict = dist_argsort[i, :topk] 
        _sim = sim_mat[i, :]    
        match_num = np.sum(_sim[serchimgdict] == 1)
        
        precX[i] = (np.float(match_num) / topk)
    return round(np.mean(precX), 3) 

def get_topk_precise_frommat(data_path):
    
    path_query_label = os.path.join(data_path, "sat_4_test_lable.mat")
    path_db_label    = os.path.join(data_path, "sat_4_train_lable.mat")
    path_query_data  = os.path.join(data_path, "sat_4_test.mat")
    path_db_data     = os.path.join(data_path, "sat_4_train.mat")
    
    query_label = loadmat(path_query_label)["sat_4_test_lable"][:]
    db_label    = loadmat(path_db_label)["sat_4_train_lable"][:]
    query_data  = loadmat(path_query_data)["sat_4_test"][:]
    db_data     = loadmat(path_db_data)["sat_4_train"][:]  
    
    db_data, query_data = data_preperation(db_data, query_data) 
    
    topk=5000
    precX  = get_topk_precise(db_data, query_data, query_label, db_label, topk)
    return precX
if __name__ == '__main__':


    path = "./logs/test_lantent_z___/"
    
    import os
    folderlist = os.listdir(path) 
    
    folderlist = ['CIFAR10Pair_Paired_Transform_0_299[1, 0, 1, 1, 0.5]64', \
        'CIFAR10Pair_Cifar_32_0_299[1, 0, 1, 1, 0.5]32',
        'CIFAR10Pair_Cifar_16_0_299[1, 0, 1, 1, 0.5]16',
        'mscoco_paired_COCO_64_0_299[1, 0, 1, 1, 0.5]64',
        'mscoco_paired_COCO_16_0_299[1, 0, 1, 1, 0.5]16',
        'mscoco_paired_COCO_32_0_299[1, 0, 1, 1, 0.5]32']
    folderlist = ['CIFAR10Pair_Hash_Contra_IR300_600[1, 0, 1, 1, 0.5]64']
    for folderiter in folderlist: 
        sonfolder = os.path.join(path, folderiter) 
        if sonfolder[-5:] == ".ckpt":
            continue
        
        epochs = os.listdir(sonfolder)
        #epochs = ['epoch_259', 'epoch_279', 'epoch_299']
        epochs = ['epoch_159']
        for ep in epochs:
            pr = os.path.join(sonfolder, ep) + "/"  
            if ep[-4:] == ".png":
                continue 
            precX  = get_topk_precise_frommat(pr) 
            print(' path = {}/{} precX = {} '.format( folderiter, ep, precX))

    print("end")
 