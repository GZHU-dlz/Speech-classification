import pandas as pd
import numpy as np
#从源文件中筛选类别对应的id
def select_covid_status(covid_id_path):
    # 使用pandas读取csv文件
    file1=pd.read_csv('E://private//Coswara-Data//combined_data.csv')
    # 转化格式，pd读取出来是DataFrame格式，转换为numpy数组格式
    file1=np.array(file1)
    
    data=[]
    for item in file1:
        sh= item[2] #每一行的第三列
        if "positive_moderate"==sh or "positive_mild" == sh or'positive_asymp' == sh:
        # if "positive_moderate" == sh or "positive_mild" == sh:    
            data.append(item[0]) 
            

    with open(covid_id_path,'w') as p:
            for path in data:
                p.write(path + '\n')
            print(f"Written {len(data)} IDs to {covid_id_path}")

def select_quality(covid_id_path):
    file_path = covid_id_path
    # 使用pandas读取csv文件
    file=pd.read_csv('/private/Coswara-Data/annotations/vowel-o_labels.csv')
    # 转化格式，pd读取出来是DataFrame格式，转换为numpy数组格式
    file=np.array(file)

    id=[]
    select_id = []
    lines = ''
    with open(file_path, 'r') as f: # 打开文件，将其值赋予file_to_read
        for line in f:
            line = line.strip()
            # print(line)
            id.append(line)

    for i in id:
        # print(i)
        for item in file:
            sh = item[0]
            quality = item[1]
            sh = sh.split('_')[0]
            # print(sh)
            if i == sh and quality != 0:
                select_id.append(i)
                # print(i)
    with open('/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_vowel_o_id.scp','w') as p:
            for path in select_id:
                p.write(path + '\n')

def get_path():
    all_path_file = '/private/Coswara-Data/coswara.scp'
    selected_quality_file_path = '/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_vowel_o_id.scp'
    select_id = []
    select_path = []
    with open(selected_quality_file_path, 'r') as f: # 打开文件，将其值赋予file_to_read
        for line in f:
            line = line.strip()
            # print(line)
            select_id.append(line)
    with open(all_path_file, 'r') as f:
          for line in f:
               line = line.strip()
               lines = line.split('/')
           
            #    print(lines[6].split('.')[0])
               for i in select_id:
                    if i == lines[5] and lines[6].split('.')[0] == 'vowel-o':
                         select_path.append(line)
    with open('/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_vowel_o_path.scp','w') as p:
            for path in select_path:
                p.write(path + '\n')

    

if __name__=="__main__":
    covid_id_path = '/private/Coswara-Data/positive_quality12_id_scp/positive_vowel_o_id.scp'
    select_covid_status(covid_id_path)
    select_quality(covid_id_path)
    get_path()

