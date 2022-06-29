from torchvision.datasets import MNIST
import numpy as np


def get_data_stats():
    dataroot = "D:/data/mode_seeking/mnist_tcr"

    train = MNIST(dataroot, train=True)
    test = MNIST(dataroot, train=False)

    train_indices = [i for i, x in enumerate((train.targets == 3) | (train.targets == 8)) if x]
    test_indices = [i for i, x in enumerate((test.targets == 3) | (test.targets == 8)) if x]
    X = np.concatenate([train.train_data[train_indices], test.test_data[test_indices]], axis=0)
    X = X.reshape((-1, 1)) / 255.
    print(f"mean(X)={X.mean()}, std(X)={X.std()}")

    return len(X)

if __name__ == "__main__":
    get_data_stats()