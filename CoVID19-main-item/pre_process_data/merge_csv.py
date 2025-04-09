import pandas as pd
import os
# 9个文件路径列表
files = [
    'E:/pkg_all/breathing_deep_quality12_5s_all_data_delete_amp_lt200_csv/breathing_deep_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/breathing_shallow_quality12_5s_all_data_delete_amp_lt200_csv/breathing_shallow_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/cough_heavy_quality12_5s_all_data_delete_amp_lt200_csv/cough_heavy_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/cough_shallow_quality12_5s_all_data_delete_amp_lt200_csv/cough_shallow_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/counting_fast_quality12_5s_all_data_delete_amp_lt200_csv/counting_fast_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/counting_normal_quality12_5s_all_data_delete_amp_lt200_csv/counting_normal_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/vowel_a_quality12_5s_all_data_delete_amp_lt200_csv/vowel_a_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/vowel_e_quality12_5s_all_data_delete_amp_lt200_csv/vowel_e_quality12_5s_all_data_delete_amp_lt200_part_1234.csv',
    'E:/pkg_all/speech_quality12_5s_all_data_delete_amp_lt200_csv/speech_quality12_5s_all_data_delete_amp_lt200_part_1234.csv'
]

output_file = 'E:/pkg_all/merged_csv_part1234.csv'  # 输出的合并文件名

# 用于存放所有数据的列表
merged_data = []

# 遍历每个文件路径
for file_path in files:
    # 检查文件是否存在
    if os.path.isfile(file_path):
        # 读取CSV文件
        try:
            data = pd.read_csv(file_path)
            merged_data.append(data)
            print(f"成功读取文件: {file_path}")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
            continue
    else:
        print(f"文件 {file_path} 不存在，跳过该文件。")

# 将所有数据拼接成一个DataFrame
if merged_data:
    final_data = pd.concat(merged_data, ignore_index=True)
    
    # 将拼接后的数据保存为新的CSV文件
    final_data.to_csv(output_file, index=False)
    print(f"合并后的数据已保存到 {output_file}")
else:
    print("没有找到CSV文件或所有文件读取失败。")