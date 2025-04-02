import csv
import os
import sys

csvFile = open("/private/Coswara-Data/cut_5s_data/quality12_5s_all_data_delete_amp_lt200.csv",'w',newline='',encoding='utf-8')
header = ('filename', 'covid_class', 'label', 'class_name', 'path')
writer = csv.writer(csvFile)
writer.writerow(header)
csvRow = []
 
f = open("/private/Coswara-Data/cut_5s_data/quality12_5s_all_data_delete_amp_lt200.scp",'r')
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)
 
f.close()
csvFile.close()