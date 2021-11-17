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
def get_precision_recall_by_Hamming_Radius(database, query, database_labels, query_labels, radius=2):
    database_output, query_output = data_preperation(database, query)  
 
    bit_n = query_output.shape[1]
    ips = compute_hamming_dist(query_output, database_output) 
    #ips = (bit_n - ips) / 2
    #ips = ips / 2
    sim_mat = gen_sim_mat(query_labels, database_labels)
    precX = [] 

    for i in range(ips.shape[0]): 
        site = ips[i, :] <= radius
        match_idx = np.reshape(np.argwhere(site), (-1))
        all_num = len(match_idx)

        if all_num != 0:
            _sim = sim_mat[i, :] 
            allsim = np.sum(_sim)
            match_num = np.sum(_sim[match_idx] == 1)
            precX.append(np.float(match_num) / all_num)
 
        else:
            precX.append(np.float(0.0))  
    return np.mean(np.array(precX))

def data_preperation(train_data, test_data):
    ''' Transfer data to binary form. '''    
    hashes_db = np.concatenate((train_data, test_data), axis=0)  
    hashes_db = hashes_db - np.dot(np.ones([hashes_db.shape[0], 1]), np.mean(hashes_db, axis = 0, keepdims = True))
    binary = np.zeros(hashes_db.shape, dtype = np.float64)
    binary[hashes_db >= 0] = 1 
    return binary[0:train_data.shape[0],:], binary[train_data.shape[0]:,:]
def cal_pr(root_path):
    
    path_query_label = os.path.join(root_path, "sat_4_test_lable.mat")
    path_db_label    = os.path.join(root_path, "sat_4_train_lable.mat")
    path_query_data  = os.path.join(root_path, "sat_4_test.mat")
    path_db_data     = os.path.join(root_path, "sat_4_train.mat")
    
    query_num = 1000
    db_num = 10000
    
    query_label = loadmat(path_query_label)["sat_4_test_lable"][:]
    db_label = loadmat(path_db_label)["sat_4_train_lable"][:]
    query_data = loadmat(path_query_data)["sat_4_test"][:]
    db_data = loadmat(path_db_data)["sat_4_train"][:]
    
    #trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy()
    #rF, qF, rL, qL, draw_range=draw_range
     
    precX = get_precision_recall_by_Hamming_Radius(db_data, query_data, db_label, query_label)
    #map = CalcTopMap(db_data, query_data, db_label, query_label, 1000)
 
    print(precX)  
    
if __name__ == '__main__':
    data_path = './data/'
    
    folderlist = ['CIFAR10Pair_Paired_Transform_0_299[1, 0, 1, 1, 0.5]64', \
        'CIFAR10Pair_Cifar_16_0_299[1, 0, 1, 1, 0.5]16', \
            'CIFAR10Pair_Cifar_32_0_299[1, 0, 1, 1, 0.5]32'] 

    path = "./logs/test_lantent_z/"
    for folder in folderlist:
        folder = os.path.join(path, folder)
        root = os.path.join(folder, 'epoch_199')
        cal_pr(root)