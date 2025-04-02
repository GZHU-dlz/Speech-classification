import torch
import numpy as np
import os
from torch.utils.data import  Dataset,DataLoader


def generate_scp_dataset(dataset_dir):
    with open('Train_Scp.txt','a',encoding='utf-8' ) as txtf :
        for dirname,subdirs,files in os.walk(dataset_dir):
            for f in files:
                if f.split('.')[-1] == 'npy':
                    txtf.write(os.path.join(dirname,f) + "\n")
    print("写入表单")

def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def segment(x, seglen=128):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:,r:r+seglen]
    return y

class MeldataSet_1(Dataset):
    def __init__(self,scp_dir,seglen):
        self.scripts = []
        self.seglen = seglen
        with open(scp_dir,encoding='utf-8') as f :
            for l in f.readlines():
                self.scripts.append(l.strip('\n'))
        self.L = len((self.scripts))
        pass
    def __getitem__(self,index):

        src_path = self.scripts[index]
        src_mel = np.load(src_path)
        return torch.FloatTensor(src_mel) ## 【80,256】

    def __len__(self):
        return self.L
        pass
    
def my_collection_way(batch):
    print("Dataloder 中的 collection func 调用：")
    print([  x.shape for x in batch])
    output = torch.stack([ torch.FloatTensor(segment(x,seglen=256)) for x in batch  ],dim=0)
    return output



    pass
if __name__ == '__main__':
    Mdata = MeldataSet_1("Train_Scp.txt",seglen=256)

    Mdataloader = DataLoader(Mdata, batch_size=3,collate_fn=my_collection_way)
    for batch in Mdataloader:
        print(batch.shape)
        exit()
