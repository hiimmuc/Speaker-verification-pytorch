import os

from matplotlib import pyplot as plt


def plot_graph(data, x_label, y_label, title, save_path, color='b-', mono=True, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    if mono:
        plt.plot(data, color=color)
    else:
        for dt in data:
            plt.plot(dt)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()


with open('exp/ResNetSE34V2/result/scores.txt') as f:
    line_data = f.readlines()

line_data = [line.strip().replace('\n', '').split(',') for line in line_data]
data = []
for line in line_data:
    if 'IT' in line[0]:
        data.append(line)
data_loss = [float(line[3].strip().split(' ')[1]) for line in data]
plot_graph(data_loss, 'epoch', 'loss', 'Loss',
           'exp/ResNetSE34V2/result/loss.png', color='b', mono=True)
data_acc = [float(line[2].strip().split(' ')[1]) for line in data]
plot_graph(data_acc, 'epoch', 'accuracy', 'Accuracy',
           'exp/ResNetSE34V2/result/acc.png', color='r')
