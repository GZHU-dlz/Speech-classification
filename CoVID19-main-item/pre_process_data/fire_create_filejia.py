import os

# 分类名称
category = "breathing_deep"
base_path = "/private/Coswara-Data"

# 要创建的路径列表，区分目录和文件
directories = [
    f"{base_path}/positive_quality12_id_scp",
    f"{base_path}/healthy_quality12_id_scp",
    f"{base_path}/cut_5s_data/positive_data",
    f"{base_path}/cut_5s_data/positive_data/{category}",
    f"{base_path}/cut_5s_data/positive_data/path_scp_{category}",
    f"{base_path}/cut_5s_data/healthy_data",
    f"{base_path}/cut_5s_data/healthy_data/{category}",
    f"{base_path}/cut_5s_data/healthy_data/path_scp_{category}",
    f"{base_path}/cut_5s_data/positive_data_after",
    f"{base_path}/cut_5s_data/healthy_data_after",
    f"{base_path}/cut_5s_data/{category}_quality12_5s_all_data_delete_amp_lt200_csv"
]

files = [
    f"{base_path}/positive_quality12_id_scp/positive_{category}_id.scp",
    f"{base_path}/positive_quality12_id_scp/positive_quality12_{category}_id.scp",
    f"{base_path}/positive_quality12_id_scp/positive_quality12_{category}_path.scp",
    f"{base_path}/healthy_quality12_id_scp/healthy_{category}_id.scp",
    f"{base_path}/healthy_quality12_id_scp/healthy_quality12_{category}_id.scp",
    f"{base_path}/healthy_quality12_id_scp/healthy_quality12_{category}_path.scp",
    f"{base_path}/cut_5s_data/positive_data/path_scp_{category}/positive_quality12_5s_{category}_path.scp",
    f"{base_path}/cut_5s_data/healthy_data/path_scp_{category}/healthy_quality12_5s_{category}_path.scp",
    f"{base_path}/cut_5s_data/positive_data_after/{category}_positive_quality12_5s_lt_1s_data_path.scp",
    f"{base_path}/cut_5s_data/healthy_data_after/{category}_healthy_quality12_5s_lt_1s_data_path.scp",
    f"{base_path}/cut_5s_data/{category}_quality12_5s_all_data_delete_amp_lt200.csv",
    
]

# 创建目录
for directory in directories:
    os.makedirs(directory, exist_ok=True)

for file_path in files:
    print(f"File path prepared: {file_path}")

print("All directories have been created successfully, and file paths have been prepared.")