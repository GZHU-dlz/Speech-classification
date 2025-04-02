import pandas as pd
import numpy as np

def select_covid_status(covid_id_path):
    file1=pd.read_csv(r'/private/Coswara-Data/combined_data.csv')
    file1=np.array(file1)

    data=[]
    for item in file1:
        sh= item[2]
        if "positive_moderate" == sh or "positive_mild" == sh:
            data.append(item[0]) 

    with open(covid_id_path,'w') as p:
            for path in data:
                p.write(path + '\n')

def select_quality(covid_id_path):
    file_path = covid_id_path
    file=pd.read_csv(r'/private/Coswara-Data/annotations/vowel-o_labels.csv')
    file=np.array(file)

    id=[]
    select_id = []
    lines = ''
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            id.append(line)

    for i in id:
        for item in file:
            sh = item[0]
            quality = item[1]
            sh = sh.split('_')[0]
            if i == sh and quality != 0:
                select_id.append(i)
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
            select_id.append(line)
    with open(all_path_file, 'r') as f:
          for line in f:
               line = line.strip()
               lines = line.split('/')
               print(lines[6].split('.')[0])
               for i in select_id:
                    if i == lines[5] and lines[6].split('.')[0] == 'vowel-o':
                         select_path.append(line)
    with open('/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_vowel_o_path.scp','w') as p:
            for path in select_path:
                p.write(path + '\n')

    

if __name__=="__main__":
    covid_id_path = '/private/Coswara-Data/positive_quality12_id_scp/positive_id.scp'
    select_quality(covid_id_path)
    get_path()
