
import os
import argparse
 
class_name = 'vowel_e'
 
def listpath(path, suffix, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listpath(file_path, suffix, list_name)
        elif os.path.splitext(file_path)[1] == suffix:
            list_name.append(file_path)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/private/Coswara-Data/cut_5s_data/positive_data/{}'.format(class_name))
    parser.add_argument("--suffix", type=str, default='.wav')
    args = parser.parse_args()
 
    list_name = []
    listpath(args.path, args.suffix, list_name)
 
    # 在写入文件之前，确保所有路径都使用正斜杠
    corrected_list_name = [path.replace('\\', '/') for path in list_name]
 
    with open('/private/Coswara-Data/cut_5s_data/positive_data/path_scp_vowel_e/positive_quality12_5s_{}_path.scp'.format(class_name), 'w') as p:
        for path in corrected_list_name:
            p.write(path + '\n')