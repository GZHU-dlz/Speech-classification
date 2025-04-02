import pandas as pd

def extract_rows_with_label(csv_file, label_value, output_file):
    df = pd.read_csv(csv_file)
    cough_df = df[df['class_name'] == label_value]
    cough_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_csv_file = "/private/Coswara-Data/cut_5s_data/quality12_5s_test0.2_data_delete_amp_lt200.csv"  # 替换为你的输入CSV文件路径
    label_to_extract = "vowel_o"
    output_csv_file = "/private/Coswara-Data/cut_5s_data/inference_csv_each_class/{}_quality12_5s_test0.2_data_delete_amp_lt200.csv".format(label_to_extract)  # 输出"Cough"标签的行保存为"cough.csv"
    

    extract_rows_with_label(input_csv_file, label_to_extract, output_csv_file)
