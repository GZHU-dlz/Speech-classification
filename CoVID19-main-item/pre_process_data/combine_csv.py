# 导入pandas库
import pandas as pd

# 读取四个csv文件，假设它们都在data文件夹下，并且都有相同的表头
df1 = pd.read_csv("/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_1.csv")
df2 = pd.read_csv("/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_2.csv")
df3 = pd.read_csv("/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_3.csv")
df4 = pd.read_csv("/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_4.csv")

df = pd.concat([df1, df2, df3, df4], axis=0)
df = df.drop_duplicates()

# 将合并后的数据框保存为新的csv文件
df.to_csv("/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_1234.csv", index=False)


