
import csv
import numpy as np
import random
import os

# 定义源文件和目标文件夹
source_file = '/private/Coswara-Data/cut_5s_data/counting_fast.csv'
dest_folder = '/private/Coswara-Data/cut_5s_data/counting_fast_csv'
split_file_prefix = "counting_fast_part_"

# 确保目标文件夹存在
os.makedirs(dest_folder, exist_ok=True)

# 读取源文件并转换为numpy数组
with open(source_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # 读取表头
    data = list(reader)    # 读取数据

# 将数据转换为numpy数组，并只保留path和label两列
data = np.array(data)
data = data[:, :2]  # 只保留前两列

# 随机打乱数据
random.shuffle(data)

# 定义每个文件的记录数（尽可能平均分配，但最后一个文件可能少一些）
num_splits = 5
records_per_split = len(data) // num_splits
remainder = len(data) % num_splits

# 拆分数据并写入新的CSV文件
for i in range(num_splits):
    start_idx = i * records_per_split
    end_idx = start_idx + records_per_split + (1 if i < remainder else 0)
    split_data = data[start_idx:end_idx]
    
    dest_file = os.path.join(dest_folder, f"{split_file_prefix}{i+1}.csv")
    with open(dest_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # 写入表头
        writer.writerows(split_data)  # 写入数据

print("Done!")