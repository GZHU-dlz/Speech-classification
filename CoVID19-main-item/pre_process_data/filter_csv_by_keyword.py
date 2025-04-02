import csv

class_label = 'test0.2'

with open('/private/Coswara-Data/cut_5s_data/quality12_5s_{}_data_delete_amp_lt200.csv'.format(class_label), 'r') as origin_csv:
    reader = csv.reader(origin_csv)
    header = next(reader)
    results = filter(lambda row: any(keyword in row[0] for keyword in ['counting', 'vowel-a', 'vowel-e', 'vowel-o']), reader)
    print(results)

    with open('/private/Coswara-Data/cut_5s_data/speech_quality12_5s_{}_data.csv'.format(class_label), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        print(results)
        writer.writerows(results)