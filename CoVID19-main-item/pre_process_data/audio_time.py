'''
统计音频时长的脚本
'''

import wave
import contextlib
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

wav_path_scp = '/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_vowel_o_path.scp'
wav_path = []
with open(wav_path_scp,'r') as f:
    for line in f:
        line = line.strip()
        print(line)
        wav_path.append(line)


audio_duration = np.arange(1,22,1)
audio_num = [0] * 21
audio_dur_20 = 0
total_time = 0

for file in wav_path:
    with contextlib.closing(wave.open(file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print("duration = ",duration)
        for i in range(len(audio_duration)-1):
            # print(f'i:{i}')
            if duration <= audio_duration[i]:
                audio_num[i] += 1
                break
            if duration > 20:
                audio_num[20] += 1
                audio_dur_20 += 1
                print(f'file_path:{file},duration:{duration}')
                break

wav_path_len = len(wav_path)
fig, ax = plt.subplots(figsize=(21,8))
bars1 = plt.bar(np.arange(1,22,1), audio_num, align='center', alpha=0.5, tick_label=audio_duration)
for b in bars1:
  height = b.get_height()
  ax.annotate('{}'.format(height),
        xy=(b.get_x() + b.get_width() / 2, height), 
        xytext=(0,3),
        textcoords="offset points",
        va = 'bottom', ha = 'center'
        )
plt.title('positive_quality12_vowel_o_num{}'.format(wav_path_len))
plt.savefig('/private/Coswara-Data/positive_quality12_id_scp/audio_time_analyse_result/positive_quality12_vowel_o.png')
plt.show()
print(f'audio_num:{audio_num}')
audio_sum = 0
for i in audio_num:
    audio_sum += i 
print(f'audio_sum:{audio_sum}')
print("总的语音段数 = ",len(wav_path))