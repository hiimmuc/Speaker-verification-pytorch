import argparse
import os

from matplotlib import pyplot as plt


def plot_graph(data, x_label, y_label, title, save_path, show=True, color='b-', mono=True, figsize=(10, 6)):
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
    if show:
        plt.show()


def plot_from_file(model, show=False):
    model_name = model
    with open(f"exp/{model_name}/result/scores.txt") as f:
        line_data = f.readlines()

    line_data = [line.strip().replace('\n', '').split(',')
                 for line in line_data]
    data = [{}]
    last_epoch = 1
    step = 10
    for line in line_data:
        if 'IT' in line[0]:
            epoch = int(line[0].split(' ')[-1])

            if epoch not in range(last_epoch - step, last_epoch + 2):
                data.append({})

            data[-1][epoch] = line
            last_epoch = epoch
    # print(data)
    for i, dt in enumerate(data):
        data_loss = [float(line[3].strip().split(' ')[1])
                     for _, line in dt.items()]
        plot_graph(data_loss, 'epoch', 'loss', 'Loss',
                   f"exp/{model_name}/result/loss_{i}.png", color='b', mono=True, show=show)
        data_acc = [float(line[2].strip().split(' ')[1])
                    for _, line in dt.items()]
        plot_graph(data_acc, 'epoch', 'accuracy', 'Accuracy',
                   f"exp/{model_name}/result/acc_{i}.png", color='r', show=show)
        plt.close()


parser = argparse.ArgumentParser(description="plot model")
if __name__ == '__main__':
    parser.add_argument('--model', type=str,
                        default='ResNetSE34V2', help='model name')
    args = parser.parse_args()
    model_name = args.model
    plot_from_file(model_name, show=False)
