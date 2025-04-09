
# 定义SCP文件的路径
positive_scp_path = '/private/Coswara-Data/cut_5s_data/label0_wav_scp/amp_label0_counting_fast_data_path.scp'
negative_scp_path = '/private/Coswara-Data/cut_5s_data/label1_wav_scp/amp_label1_counting_fast_data_path.scp'
third_scp_path = '/private/Coswara-Data/cut_5s_data/label2_wav_scp/amp_label2_counting_fast_data_path.scp'

# 定义新CSV文件的路径
csv_output_path = '/private/Coswara-Data/cut_5s_data/counting_fast.csv'

# 读取SCP文件并获取路径列表
with open(positive_scp_path, 'r') as f:
    positive_paths = [line.strip() for line in f.readlines()]

with open(negative_scp_path, 'r') as f:
    negative_paths = [line.strip() for line in f.readlines()]

with open(third_scp_path, 'r') as f:
    third_paths = [line.strip() for line in f.readlines()]

# 创建一个列表来存储所有样本，每个样本是一个字典，包含路径和标签
all_samples = [
    {'path': path, 'label': 1} for path in positive_paths  # 第一个路径的标签为1
] + [
    {'path': path, 'label': 0} for path in negative_paths  # 第二个路径的标签为0
] + [
    {'path': path, 'label': 2} for path in third_paths     # 第三个路径的标签为2
]

# 打乱列表中的样本顺序
random.shuffle(all_samples)

# 写入CSV文件
with open(csv_output_path, 'w', newline='') as csvfile:
    fieldnames = ['path', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # 写入所有打乱顺序的样本
    for sample in all_samples:
        writer.writerow(sample)

print(f"CSV文件已生成并保存到 {csv_output_path}，样本顺序已打乱")
