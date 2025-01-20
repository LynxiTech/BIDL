# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import csv
import os
from PIL import Image

# Define CSV file column names
headers = ['video', 'class', 'label', 'frames', 'height','width']


# Specify the path to the CSV file
#csv_file_path_datas = 'jester-v1-train.csv' 
csv_file_path_datas = 'jester-v1-validation.csv' 
# Use a list to store the data
datas = []
 

with open(csv_file_path_datas, mode='r', newline='') as file:
    reader = csv.reader(file)
    # Iterate through each row in the CSV file
    for row in reader:
        # Add each row as a list to datas
        split_list = row[0].split(';')
        datas.append(split_list)

csv_file_path_labels = 'jester-v1-labels.csv'
 
# Use a list to store the label
labels_list = []
labels = {}

with open(csv_file_path_labels, mode='r', newline='') as file:
    reader = csv.reader(file)
    i=0
    for row in reader:
        labels_list.append(row[0])
    labels_list.sort(key=str.lower)
    for label in labels_list:
        if label in labels:
            pass
        else:
            labels[label]=i
            i+=1

# Define the data rows to write to the CSV file
pred_data=[]
for data in datas:
    video=data[0]
    class_name=data[1]
    class_label=labels[class_name]

    target_folder = 'frames/'+video+'/'
    file_names = sorted(os.listdir(target_folder))
    frames = ""
    for file_name in file_names:
        if file_name.lower().endswith('.jpg'):
            try:
                with Image.open(target_folder+file_name) as img:
                    width, height = img.size
                    if frames:
                        frames += "|"
                    frames += file_name
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    pred_data.append([video,class_name,class_label,frames,height,width])
        

# Specify the path to the CSV file
#csv_file_path = 'train.csv'
csv_file_path = 'val.csv'


with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the column names
    writer.writerow(headers)
    # Write the data rows
    writer.writerows(pred_data)

print(f"CSV file has been created and written to {csv_file_path}")