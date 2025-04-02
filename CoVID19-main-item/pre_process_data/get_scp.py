'''
获取音频路径的scp
'''

import os 
import argparse

class_name = 'counting_normal'
def listpath(path, suffix, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if os.path.isdir(file_path):
            listpath(file_path, suffix, list_name)
        elif os.path.splitext(file_path)[1] == suffix:
            # list_name.append(file.split(suffix)[0] + ' ' + file_path)
            list_name.append(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/private/Coswara-Data/cut_5s_data/positive_data/{}'.format(class_name))
    parser.add_argument("--suffix", type=str, default='.wav')
    args = parser.parse_args()

    list_name = []
    listpath(args.path, args.suffix, list_name)
    with open('/private/Coswara-Data/cut_5s_data/positive_data/path_scp/positive_quality12_5s_{}_path.scp'.format(class_name),'w') as p:
        for path in list_name:
            p.write(path + '\n')