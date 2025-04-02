import pandas as pd

def merge_csv_files(file1, file2, file3, file4, file5, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)
    merged_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_csv_file1 = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/counting_fast_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为第一个CSV文件路径
    input_csv_file2 = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/counting_normal_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为第二个CSV文件路径
    input_csv_file3 = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/vowel_a_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为第二个CSV文件路径
    input_csv_file4 = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/vowel_e_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为第二个CSV文件路径
    input_csv_file5 = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/vowel_o_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为第二个CSV文件路径
    output_csv_file = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/speech_quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 合并后的输出文件名

    merge_csv_files(input_csv_file1, input_csv_file2, input_csv_file3, input_csv_file4, input_csv_file5, output_csv_file)
