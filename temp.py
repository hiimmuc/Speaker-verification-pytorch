import os

parent_dir = 'dataset/public_test/data_test'
with open('dataset/public_test/data_public_test.txt', 'r') as f:
    data = f.readlines()
# print(data[:10])
data = [list(line.split(' ')) for line in data]
# print(data[:3])
for i in range(len(data)):
    data[i][0] = os.path.join(parent_dir, data[i][0])
    data[i][1] = os.path.join(parent_dir, data[i][1])
data = [f'{data[i][0]} {data[i][1]}' for i in range(len(data))]
with open('dataset/data_public_test.txt', 'w') as f:
    f.writelines(data)
# print(data[:3])
print('Done')
