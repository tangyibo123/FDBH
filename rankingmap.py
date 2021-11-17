import numpy as np 
from scipy.io import loadmat

def hamming_distance(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    p1 = np.sign(b1).astype(np.int8)
    p2 = np.sign(b2).astype(np.int8)

    r = p1.shape[1]
    d = (r - np.matmul(p1, np.transpose(p2))) // 2
    return d

def hamming_rank(b1, b2):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    dist_h = hamming_distance(b1, b2)
    return np.argsort(dist_h, 1, kind='mergesort')

def compute_map_from_hashes(hashes_db, hashes_query, labels_db, labels_query, top_n=0):
    """Compute MAP for given set of hashes and labels"""
    rank = hamming_rank(hashes_query, hashes_db)
    s = _compute_similarity(labels_db, labels_query)
    return compute_map_from_rank(rank, s, top_n)

def _compute_similarity(labels_db, labels_query, and_mode=True):
    """Return similarity matrix between two label vectors.
    The output is binary matrix of size n_train x n_test.
    """
    if and_mode:
        labels_db = labels_db.astype(dtype=np.bool)
        labels_query =  labels_query.astype(dtype=np.bool)
        return np.sum(np.bitwise_and(labels_db, labels_query[:, np.newaxis]), axis = -1).astype(dtype=np.bool)
        #return np.bitwise_and(labels_db.astype(dtype=np.bool), np.transpose(labels_query.astype(dtype=np.bool))).astype(dtype=np.bool)
    else: 
        return np.equal(labels_db, labels_query[:, np.newaxis])[:,:,0]
def compute_map(rank, s, top_n):
    Q, N = s.shape
    if top_n == 0:
        top_n = N
    pos = np.asarray(range(1, top_n + 1), dtype=np.float32)
    mAP = 0
    av_precision = np.zeros(top_n)
    av_recall = np.zeros(top_n)

    for q in range(Q): 
        relevance = s[q, rank[q]]
        relevance = relevance[ : top_n]

        ap = 0
        for idx in range(len(relevance)):
            if relevance[idx] is True:
                ap += ap / idx



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

def eval_cls_map(query, target, cls1, cls2, at=None):
    """
    Mean average precision computation
    :param query:
    :param target:
    :param cls1:
    :param cls2:
    :param at:
    :return:
    """
    serchimgdict = {}
    serchrightdict = {}
    sim_mat = gen_sim_mat(cls1, cls2)
    query_size = query.shape[0]
    distances = compute_hamming_dist(query, target)

    radius = 2
    
    dist_argsort = np.argsort(distances)

    map_count = 0.
    average_precision = 0.

    perap = np.zeros(query_size)
    for i in range(query_size): 
        gt_count = 0.
        precision = 0.

        top_k = at if at is not None else dist_argsort.shape[1]
        serchimgdict[i] = dist_argsort[i, :top_k] 
        serchrightdict[i] = np.ones((top_k,))
        for j in range(top_k):
            this_ind = dist_argsort[i, j]
            if sim_mat[i, this_ind] == 1:
                serchrightdict[i][j] = 1
                gt_count += 1.
                precision += gt_count / (j + 1.)
            else:
                serchrightdict[i][j] = 0
        if gt_count > 0:
            average_precision += precision / gt_count
            map_count += 1.
            perap[i] = precision / gt_count
        else: 
            map_count += 1.
    return average_precision / map_count, serchimgdict, serchrightdict, perap

def compute_map_from_rank(rank, s, top_n):
    """compute mean average precision (MAP)"""
    Q, N = s.shape
    if top_n == 0:
        top_n = N
    pos = np.asarray(range(1, top_n + 1), dtype=np.float32)
    mAP = 0
    av_precision = np.zeros(top_n)
    av_recall = np.zeros(top_n)
    for q in range(Q):
        rs = rank[q, :top_n]
        relevance = s[q, rs]
        cumulative = np.cumsum(relevance)
        max_number_of_relevant_documents = min(np.sum(s[q]), top_n)
 
        if max_number_of_relevant_documents != 0:
            precision = cumulative.astype(np.float32) / pos
            recall = cumulative / max_number_of_relevant_documents
            av_precision += precision
            av_recall += recall
            ap = np.dot(precision.astype(np.float64), relevance)
            ap /= max_number_of_relevant_documents
            mAP += ap 
    mAP /= Q
    av_precision /= Q
    av_recall /= Q

    return float(mAP), av_precision, av_recall

def data_preperation(train_data, test_data):
    ''' Transfer data to binary form. '''
    
    hashes_db = np.concatenate((train_data, test_data), axis=0)
  
    hashes_db = hashes_db - np.dot(np.ones([hashes_db.shape[0], 1]), np.mean(hashes_db, axis = 0, keepdims = True))

    binary = np.zeros(hashes_db.shape, dtype = np.float64)
    binary[hashes_db >= 0] = 1
    return binary[0:train_data.shape[0],:], binary[train_data.shape[0]:,:]

def ALLINONE(path, top_k= 10):
    path = path + "{}"
    test_lable_path = path.format("sat_4_test_lable.mat")         # the format of dictionary is like : {"sat_4_test_lable": data}
    train_lable_path = path.format("sat_4_train_lable.mat")     # the format of dictionary is like : {"sat_4_train_lable": data}
    test_data_path = path.format("sat_4_test.mat")               # the format of dictionary is like : {"sat_4_test": data}
    train_data_path = path.format("sat_4_train.mat")             # the format of dictionary is like : {"sat_4_train": data}
    test_image_path = path.format("sat_4_test_img.mat")
    train_image_path = path.format("sat_4_train_img.mat")

    # Cal mAP for top 5000 images
    
    # load mat
    # Plz remove the data length limitation if your computer has enougth memory  
    test_lable = loadmat(test_lable_path)["sat_4_test_lable"]
    test_data = loadmat(test_data_path)["sat_4_test"]
    train_lable = loadmat(train_lable_path)["sat_4_train_lable"]
    train_data = loadmat(train_data_path)["sat_4_train"]
    
    # Constant funct
    #binary_train, binary_test = train_data, test_data
    binary_train, binary_test = data_preperation(train_data, test_data)
    mAP, serchdictimg, serchdictright, perap = eval_cls_map(binary_test, binary_train, test_lable, train_lable, top_k)
 
    return mAP

if __name__ == '__main__':

    path = "./logs/test_lantent_z___/"
    path = "./logs/test_lantent_z/"
    #path = "./logs/test_lantent_z_findbesthyper/"
    import os
    folderlist = os.listdir(path) 
    folderlist = ['nuswild_paired_ALLIR[1, 0, 1, 0, 0.5]64']  
    folderlist = ["nuswild_paired_LALAYER_IR[10, 0, 1, 0, 0.5]64"]
    #folderlist = ['CIFAR10Pair_ALLIR[1, 0, 1, 0, 0.5]64']
    folderlist = ['nuswild_paired_AE[1, 0, 1, 0, 0.5]64']
    folderlist = ['CIFAR10Pair_CIFAR_[1, 0, 1, 100, 0.5]64', 'CIFAR10Pair_CIFAR_[1, 0, 1, 500, 0.5]64', 
    'CIFAR10Pair_CIFAR_[1, 0, 1, 50, 0.5]64', 'CIFAR10Pair_CIFAR_[1, 0, 1, 10, 0.5]64', 
    'CIFAR10Pair_CIFAR_[1, 0, 1, 5, 0.5]64', 'CIFAR10Pair_CIFAR_[1, 0, 1, 1, 0.5]64' ]
    folderlist = ['CIFAR10Pair_CIFAR_[1, 0, 1, 0, 0.5]64']
    folderlist= ['CIFAR10Pair_Hash_Contra_IR0_299[1, 0, 1, 1, 0.5]64']
    folderlist= ['CIFAR10Pair_Hash_Contra_IR300_600[1, 0, 1, 1, 0.5]64']
    folderlist= ['CIFAR10Pair_Hash_Contra_IR_NoneRelu_0_299[1, 0, 1, 1, 0.5]64']

    folderlist = ['mscoco_paired_COCO_16_0_299[1, 0, 1, 1, 0.5]16']
    folderlist = ['CIFAR10Pair_Paired_Transform_0_299[1, 0, 1, 1, 0.5]64', \
        'CIFAR10Pair_Cifar_16_0_299[1, 0, 1, 1, 0.5]16', \
            'CIFAR10Pair_Cifar_32_0_299[1, 0, 1, 1, 0.5]32'] 
    folderlist = ['nuswild_paired_NUS_M_64_0_299[1, 0, 1, 1, 0.5]64',\
        'nuswild_paired_NUS_M_32_0_299[1, 0, 1, 1, 0.5]32',\
            'nuswild_paired_NUS_M_16_0_299[1, 0, 1, 1, 0.5]16']
    folderlist = ['mscoco_paired_COCO_64_0_299[1, 0, 1, 1, 0.5]64', \
        'mscoco_paired_COCO_16_0_299[1, 0, 1, 1, 0.5]16', \
            'mscoco_paired_COCO_32_0_299[1, 0, 1, 1, 0.5]32']

    folderlist = ['CIFAR10Pair_withoutcontr_64[1, 0, 1, 1, 0.5]64',
    'CIFAR10Pair_withoutcontr_32[1, 0, 1, 1, 0.5]32',
    'CIFAR10Pair_withoutcontr_16[1, 0, 1, 1, 0.5]16',
    'CIFAR10Pair_withoutpolar_64[1, 0, 1, 1, 0.5]64',
    'CIFAR10Pair_withoutpolar_32[1, 0, 1, 1, 0.5]32',
    'CIFAR10Pair_withoutpolar_16[1, 0, 1, 1, 0.5]16',
    'CIFAR10Pair_withoutir_64[1, 0, 1, 1, 0.5]64',
    'CIFAR10Pair_withoutir_32[1, 0, 1, 1, 0.5]32',
    'CIFAR10Pair_withoutir_16[1, 0, 1, 1, 0.5]16',
    ]

    folderlist = [ 
    'CIFAR10Pair_Cifar_16[1, 0, 0.7, 1, 0.5]16' ,
    'CIFAR10Pair_Cifar_16[1, 0, 4, 1, 0.5]16',
    'CIFAR10Pair_Cifar_16[1, 0, 2, 1, 0.5]16',
    ]  
    rangkingscore = {}
    for folderiter in folderlist: 
        sonfolder = os.path.join(path, folderiter)
        '''
        try:
            mAP = ALLINONE(os.path.join(sonfolder, os.listdir(sonfolder)[-1]) + "/") 
        except:
            pass 
        print('path = {}  mAP = {} '.format(folderiter, mAP))
        '''
        if sonfolder[-5:] == ".ckpt":
            continue
        
        epochs = os.listdir(sonfolder)
        epochs = ['epoch_159', 'epoch_179', 'epoch_199']
        #epochs = ['epoch_199']
        for ep in epochs:
            pr = os.path.join(sonfolder, ep) + "/" 
            #pr = "/home/bbct/wangfan/code/ViT-VAE/logs/test_lantent_z/SAT-4_VitVAE[1, 1, 3]32/epoch_4/"
            if ep[-4:] == ".png":
                continue
            
            for top_k in [1000]:
                mAP = ALLINONE(pr, top_k) 
                print('top_k = {} path = {}/{} mAP = {} '.format(top_k, folderiter, ep, mAP))
            rangkingscore[folderiter + ep] = mAP
    print("end")
