import os 
import numpy as np
import random
 
root_dir = "/home/bbct/publicdatasets/database"

root_dir0 = "/home/bbct/publicdatasets/database/0"
root_dir1 = "/home/bbct/publicdatasets/database/1"
root_dir2 = "/home/bbct/publicdatasets/database/2"
root_dir3 = "/home/bbct/publicdatasets/database/3"
root_dir4 = "/home/bbct/publicdatasets/database/4"

list0 = os.listdir(root_dir0)
list1 = os.listdir(root_dir1)
list2 = os.listdir(root_dir2)
list3 = os.listdir(root_dir3)
list4 = os.listdir(root_dir4) 
ourlist = []
for item in list0:
    ourlist.append(os.path.join("./0", item)) 
for item in list1:
    ourlist.append(os.path.join("./1", item)) 
for item in list2:
    ourlist.append(os.path.join("./2", item)) 
for item in list3:
    ourlist.append(os.path.join("./3", item)) 
for item in list4:
    ourlist.append(os.path.join("./4", item)) 
 
random.shuffle(ourlist) 
length = len(ourlist)
 
train_list = ourlist[0: int(length / 10 * 6)]
test_list =  ourlist[int(length / 10 * 6): int(length / 10 * 8)] 
val_list =   ourlist[int(length / 10 * 8): ] 

def save_txt(path, list):
    with open(path, 'w') as f: 
        for values in list: 
            f.write(values + "\r")


save_txt(os.path.join(root_dir, "train.txt"), train_list)
save_txt(os.path.join(root_dir, "test.txt"), test_list)
save_txt(os.path.join(root_dir, "val.txt"), val_list)



print(len(ourlist), len(train_list), len(test_list), len(val_list))