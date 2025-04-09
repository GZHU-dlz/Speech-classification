with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing.csv", "r") as f1, open("/private/Coswara-Data/cut_5s_data/quality12_5s_amp_than200_lt_0.25sr_data_path.scp", "r") as f2:
    # 读取文件内容并按行分割
    w_lines = f1.read().splitlines()
    q_lines = f2.read().splitlines()
new_w_lines = []

for w_line in w_lines:
    w_col5 = w_line.split(',')[4]
    found = False
    for q_line in q_lines:
        if w_col5 == q_line:
            found = True
            break
    if not found:
        new_w_lines.append(w_line)

# 打开w.txt文件并以写入模式打开
with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200.csv", "w") as f1:
    # 遍历新列表并将每一行写入文件中，加上换行符
    for new_w_line in new_w_lines:
        f1.write(new_w_line + "\n")