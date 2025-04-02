# 导入模块
import csv
import numpy as np

source_file = "/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200.csv"
dest_folder = "/private/Coswara-Data/cut_5s_data/speech_quality12_5s_all_data_delete_amp_lt200_csv"
split_file_prefix = "speech_quality12_5s_all_data_delete_amp_lt200_part_"
records_per_file = 5
with open(source_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)
    data = np.array(data)


unique_labels = np.unique(data[:,2])
print(f'label:{unique_labels}')
unique_classes = np.unique(data[:,3])
print(f'class:{unique_classes}')

split_data_1 = []
split_data_2 = []
split_data_3 = []
split_data_4 = []
split_data_5 = []

for label in unique_labels:
    for class_name in unique_classes:
        sub_data = data[(data[:,2] == label) & (data[:,3] == class_name)]
        chunks = np.array_split(sub_data,5)
        split_data_1.append(chunks[0])
        split_data_2.append(chunks[1])
        split_data_3.append(chunks[2])
        split_data_4.append(chunks[3])
        split_data_5.append(chunks[4])

split_data_1 = np.concatenate(split_data_1)
split_data_2 = np.concatenate(split_data_2)
split_data_3 = np.concatenate(split_data_3)
split_data_4 = np.concatenate(split_data_4)
split_data_5 = np.concatenate(split_data_5)

names = locals()


for i in range(5):
    dest_file = dest_folder + "/" + split_file_prefix + str(i+1) + ".csv"
    with open(dest_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(names.get('split_data_' + str(i+1)))

print("Done!")