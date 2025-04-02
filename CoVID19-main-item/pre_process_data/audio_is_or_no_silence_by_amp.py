import torchaudio 
import torch
import wavio
import numpy as np
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
lt_1s = []
with open('/private/Coswara-Data/cut_5s_data/quality12_5s_all_data_path.scp') as f:
    for line in f:
        audio_file = line.strip()
        count = 0
        sig = wavio.read(audio_file)
        for i in sig.data:
            if abs(i[0]) > 500:
                count += 1
        if count >= sig.rate:
            lt_1s.append(audio_file)
with open('/private/Coswara-Data/cut_5s_data/quality12_5s_lt_1s_data_path.scp','w') as p:
    for path in lt_1s:
        p.write(path + '\n')
