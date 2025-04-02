from math import floor
csv_data = []
with open ('/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200.csv') as file:
    csv_data.append(file.read())
csv_data = (''.join(csv_data)).split("\n")
header = csv_data[0]
csv_data = csv_data[1:]
temp_list = []
add_header = True
count = 0
for i in csv_data:
    count += 1
    if len(temp_list) == 0:
        temp_list.append(i)
    elif i.split(',')[2] == temp_list[0].split(',')[2] and i.split(',')[3] == temp_list[0].split(',')[3]:
        temp_list.append(i)
        if count == len(csv_data):
            file_length = len(temp_list)
            line_count = floor((0.8*file_length)+1)
            if line_count == 1:
                with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.8.csv","a+") as file1:
                    if add_header:
                        add_header = False
                        file1.write(header+'\n')
                    file1.write(temp_list[0]+'\n')
            else:
                seventy_perc_lines = temp_list[:line_count-1]
                thirty_perc_lines = temp_list[line_count-1:]
                if add_header:
                    seventy_perc_lines.insert(0,header)
                    thirty_perc_lines.insert(0,header)
                    add_header = False
                with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.8.csv","a+") as file1:
                    for j in range(len(seventy_perc_lines)):
                        file1.write(seventy_perc_lines[j]+'\n')
                if len(thirty_perc_lines) != 0:
                    with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.8.csv","a+") as file2:
                        for j in range(len(thirty_perc_lines)):
                            file2.write(thirty_perc_lines[j]+'\n')
        else:
            pass
    else:
        file_length = len(temp_list)
        line_count = floor((0.8*file_length)+1)
        if line_count == 1:
            with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.8.csv","a+") as file1:
                if add_header:
                    add_header = False
                    file1.write(header+'\n')
                file1.write(temp_list[0]+'\n')
        else:
            seventy_perc_lines = temp_list[:line_count-1]
            thirty_perc_lines = temp_list[line_count-1:]
            if add_header:
                seventy_perc_lines.insert(0,header)
                thirty_perc_lines.insert(0,header)
                add_header = False
            with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.8.csv","a+") as file1:
                for j in range(len(seventy_perc_lines)):
                    file1.write(seventy_perc_lines[j]+'\n')
            if len(thirty_perc_lines) != 0:
                with open("/private/Coswara-Data/cut_5s_data/quality12_5s_breathing_delete_amp_lt200_train0.2.csv","a+") as file2:
                    for j in range(len(thirty_perc_lines)):
                        file2.write(thirty_perc_lines[j]+'\n')
        temp_list = []
        temp_list.append(i)