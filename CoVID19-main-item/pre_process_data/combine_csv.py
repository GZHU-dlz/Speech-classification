# 导入pandas库
import pandas as pd
# 读取四个csv文件，假设它们都在data文件夹下，并且都有相同的表头
df1 = pd.read_csv("F:/covid-19-sounds/all_wav_meta_amp_csv/all_wav_meta_amp_part_1.csv")
df2 = pd.read_csv("F:/covid-19-sounds/all_wav_meta_amp_csv/all_wav_meta_amp_part_2.csv")
df3 = pd.read_csv("F:/covid-19-sounds/all_wav_meta_amp_csv/all_wav_meta_amp_part_3.csv")
df4 = pd.read_csv("F:/covid-19-sounds/all_wav_meta_amp_csv/all_wav_meta_amp_part_4.csv")

# 将四个数据框沿着行方向合并为一个数据框
df = pd.concat([df1, df2, df3, df4], axis=0)
# 将合并后的数据框保存为新的csv文件
df.to_csv("F:/covid-19-sounds/all_wav_meta_amp_csv/all_wav_meta_amp_part_1234.csv", index=False)
