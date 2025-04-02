
import copy
import os

new = []
new_path = ''
with open(r'/private/Coswara-Data/healthy_quality12_id_scp/healthy_quality12_all_path.scp') as f:
    for path in f:
        path = path.strip()
        path = path.split("/")
        new_path = copy.deepcopy(path)
        with open(r'/private/Coswara-Data/combined_data.csv',encoding='gb18030') as v:
            for meta in v:
                meta = meta.strip()
                meta = meta.split(',')
                if path[5] == meta[0]:
                    new_path[6] = meta[1] + '_' + meta[6] + '_' + new_path[6]
                    old_path = '/'.join(path)
                    new_path = '/'.join(new_path)
                    new.append(new_path)
with open ('/private/Coswara-Data/healthy_quality12_id_scp/healthy_quality12_all_new_path.scp','w') as n:
    for new_name_path in new:
        n.write(new_name_path + '\n')

