from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import os
draw_range = [1, 500, 1000, 2000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 50000]

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

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
def pr_curve2(retrieval_code, query_code, retrieval_targets, query_targets):
    import torch
    retrieval_code = torch.Tensor(retrieval_code)
    query_code = torch.Tensor(query_code)
    retrieval_targets = torch.Tensor(retrieval_targets)
    query_targets = torch.Tensor(query_targets)
    
    device = torch.device("cpu")
    """
    P-R curve.
    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.
    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1).to(device)
    R = torch.zeros(num_query, num_bit + 1).to(device)
    for i in range(num_query):
        gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask

    return P, R
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
#trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy()
def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)

    #Gnd = gen_sim_mat(qL, rL)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(round(np.mean(p), 3))
        R.append(round(np.mean(r), 3))
    return P, R

def data_preperation(train_data, test_data):
    ''' Transfer data to binary form. '''    
    hashes_db = np.concatenate((train_data, test_data), axis=0)  
    hashes_db = hashes_db - np.dot(np.ones([hashes_db.shape[0], 1]), np.mean(hashes_db, axis = 0, keepdims = True))
    binary = np.zeros(hashes_db.shape, dtype = np.float64)
    binary[hashes_db >= 0] = 1 
    binary[hashes_db < 0] = -1 
    return binary[0:train_data.shape[0],:], binary[train_data.shape[0]:,:]

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

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
    
    db_data, query_data = data_preperation(db_data, query_data) 
    #trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy()
    #rF, qF, rL, qL, draw_range=draw_range
    P, R = pr_curve(db_data, query_data, db_label, query_label)
    #map = CalcTopMap(db_data, query_data, db_label, query_label, 1000)
    print(P)
    print(R)
    #print(map)
    
if __name__ == '__main__':
     
    
    path = "./logs/test_lantent_z___/"
    folderlist = ['CIFAR10Pair_Paired_Transform_0_299[1, 0, 1, 1, 0.5]64', \
        'CIFAR10Pair_Cifar_32_0_299[1, 0, 1, 1, 0.5]32', \
            'CIFAR10Pair_Cifar_16_0_299[1, 0, 1, 1, 0.5]16'] 
 
    path = "./logs/test_lantent_z/"
    folderlist = ['CIFAR10Pair_WithoutAE_64[1, 0, 1, 1, 0.5]64']
    for folder in folderlist:
        folder = os.path.join(path, folder)
        root = os.path.join(folder, 'epoch_199')
        cal_pr(root)