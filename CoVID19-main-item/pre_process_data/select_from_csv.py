import pandas as pd
import numpy as np
#从源文件中筛选类别对应的id
def select_covid_status(class0_id_path,class1_id_path,class2_id_path):
    # 使用pandas读取csv文件
    file1=pd.read_csv('E://private//Coswara-Data//combined_data.csv')
    # 转化格式，pd读取出来是DataFrame格式，转换为numpy数组格式
    file1=np.array(file1)
    
    data_0=[]
    data_1=[]
    data_2=[]
    for item in file1:
        sh= item[2]
        if "positive_moderate"==sh or "positive_mild" == sh or'positive_asymp' == sh:
            data_0.append(item[0])
        if "healthy"==sh or "recovered_full"== sh:
            data_1.append(item[0]) 
            

    with open(class0_id_path,'w') as p:
            for path in data_0:
                p.write(path + '\n')
            print(f"Written {len(data_0)} IDs to {class0_id_path}")
    with open(class1_id_path,'w') as p:
            for path in data_1:
                p.write(path + '\n')
            print(f"Written {len(data_1)} IDs to {class1_id_path}")
def select_quality(class0_id_path,class1_id_path,class2_id_path):
    file_path0,file_path1 = class0_id_path,class1_id_path
    file=pd.read_csv('/private/Coswara-Data/annotations/counting-fast_labels.csv')
    file=np.array(file)
    id0=[]
    id1=[]
    select_id0 = []
    select_id1=[]
    with open(file_path0, 'r') as f1: # 打开文件，将其值赋予file_to_read
        for line in f1:
            line = line.strip()
            id0.append(line)
    with open(file_path1, 'r') as f1: # 打开文件，将其值赋予file_to_read
        for line in f1:
            line = line.strip()
            id1.append(line)
    for i in id0:
        for item in file:
            sh = item[0]
            quality = item[1]
            sh = sh.split('_')[0]
            if i == sh and quality != 0:
                select_id0.append(i)
    for i in id1:
        for item in file:
            sh = item[0]
            quality = item[1]
            sh = sh.split('_')[0]
            if i == sh and quality != 0:
                select_id1.append(i)

    with open('/private/Coswara-Data/label0_quality12_id_scp/label0_quality12_counting_fast_id.scp','w') as p:
            for path in select_id0:
                p.write(path + '\n')
    with open('/private/Coswara-Data/label1_quality12_id_scp/label1_quality12_counting_fast_id.scp','w') as p:
            for path in select_id1:
                p.write(path + '\n')

def get_path():
    all_path_file = '/private/Coswara-Data/coswara.scp'
    selected_quality_file_path0 = '/private/Coswara-Data/label0_quality12_id_scp/label0_quality12_counting_fast_id.scp'
    selected_quality_file_path1 = '/private/Coswara-Data/label1_quality12_id_scp/label1_quality12_counting_fast_id.scp'
    select_id0 = []
    select_id1=[]
    select_path0 = []
    select_path1=[]
    with open(selected_quality_file_path0, 'r') as f: # 打开文件，将其值赋予file_to_read
        for line in f:
            line = line.strip()
            select_id0.append(line)
    with open(selected_quality_file_path1, 'r') as f: # 打开文件，将其值赋予file_to_read
        for line in f:
            line = line.strip()
            select_id1.append(line)
            
    with open(all_path_file, 'r') as f:
          for line in f:
               line = line.strip()
               lines = line.split('/')
               label = 'counting-fast'
               for i in select_id0:
                    if i == lines[5] and lines[6].split('.')[0] == label:
                         select_path0.append(line)
               for i in select_id1:
                    if i == lines[5] and lines[6].split('.')[0] == label:
                         select_path1.append(line)
    with open('/private/Coswara-Data/label0_quality12_id_scp/label0_quality12_counting_fast_path.scp','w') as p:
            for path in select_path0:
                p.write(path + '\n')
    with open('/private/Coswara-Data/label1_quality12_id_scp/label1_quality12_counting_fast_path.scp','w') as p:
            for path in select_path1:
                p.write(path + '\n')

    

if __name__=="__main__":
    class0_id_path = '/private/Coswara-Data/label0_quality12_id_scp/label0_counting_fast_id.scp'
    class1_id_path = '/private/Coswara-Data/label1_quality12_id_scp/label1_counting_fast_id.scp'
    select_covid_status(class0_id_path,class1_id_path)
    select_quality(class0_id_path,class1_id_path)
    get_path()

