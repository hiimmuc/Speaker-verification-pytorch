import csv
import os

from tqdm import tqdm

parent_dir = 'dataset/public_test/data_test'
with open('dataset/data_public_test.txt', 'r') as f:
    data = f.readlines()
# print(data[:10])
data = [list(line.replace('\n', '').split(' ')) for line in data]
# # print(data[:3])
# for i in range(len(data)):
#     data[i][0] = os.path.join(parent_dir, data[i][0])
#     data[i][1] = os.path.join(parent_dir, data[i][1])
# data2 = [f'{data[i][0]} {data[i][1]}\n' for i in range(len(data))]
# with open('dataset/data_public_test.txt', 'w') as f:
#     f.writelines(data2)

# data3 = []
with open('dataset/public-test.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in range(len(data)):
        writer.writerow(data[i])
# print(data[:3])
print('Done')

files = []
lines = []
with open('dataset/public-test.csv', newline='') as rf:
    spamreader = csv.reader(rf, delimiter=',')
    next(spamreader, None)
    for row in tqdm(spamreader):
        files.append(row[0])
        files.append(row[1])
        lines.append(row)
print(files[:3])
