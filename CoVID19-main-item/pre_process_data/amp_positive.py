import torchaudio 
import torch
import wavio
import numpy as np
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
lt_1s = []
with open('/private/Coswara-Data/cut_5s_data/positive_data/path_scp_vowel_a/positive_quality12_5s_vowel_a_path.scp') as f:
    for line in f:
        audio_file = line.strip()
        count = 0
        sig = wavio.read(audio_file)
        for i in sig.data:
            if abs(i[0]) > 500:
                count += 1
        if count >= sig.rate:
            lt_1s.append(audio_file)
            print(audio_file)
corrected_list_name = [path.replace('\\', '/') for path in lt_1s]
with open('/private/Coswara-Data/cut_5s_data/positive_data_after/vowel_a_positive_quality12_5s_lt_1s_data_path.scp','w') as p:
    for path in corrected_list_name:
        p.write(path + '\n')
        print(path)
