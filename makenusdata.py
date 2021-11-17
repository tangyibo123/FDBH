import numpy as np
imglistdb = open("./nus_wide_m/database.txt").readlines()
imglistte = open("./nus_wide_m/train.txt").readlines()
imglisttr = open("./nus_wide_m/test.txt").readlines()

dict= {}

def mkdict(imglist, dict):
    for val in imglist:
        content = val[25:]  
        imgname = content.split()[0]
        label = np.array([int(la) for la in content.split()[1:]])
        #if imgname not in dict.keys(): 
        dict[imgname] = label 

mkdict(imglistdb, dict)
mkdict(imglistte, dict)
mkdict(imglisttr, dict)

print(len(dict))

print(dict['27863_809224894_90ee058136_m.jpg'])
srcimglisttr = open("./nus_wide_m/test_21.txt").readlines()
#imglist = [imgpath[7:] for imgpath in imglistdb]
print(srcimglisttr[0])
print(srcimglisttr[0][7:])


srcimglistdb = open("./nus_wide_m/database_21.txt").readlines()
srcimglistte = open("./nus_wide_m/train_21.txt").readlines()
srcimglisttr = open("./nus_wide_m/test_21.txt").readlines()
imglist = [imgpath[25:] for imgpath in imglistdb]
srcdictdb= {}
srcdicttr= {}
srcdictte= {}

def mksrcdict(imglist, srcdict, dict):
    for val in imglist:

        imgname = val.split()[0][7:]
        imgpath = val.split()[0]
        if imgname not in srcdict.keys(): 
            srcdict[imgpath] = dict[imgname] 

mksrcdict(srcimglistdb, srcdictdb, dict)
mksrcdict(srcimglistte, srcdicttr, dict)
mksrcdict(srcimglisttr, srcdictte, dict)

print(len(dict))
