import os


def func():
    # try:
    #     os.system("python cifar10_regular.py")
    # except Exception as e:
    #     pass
    try:
        os.system("python cifar10_large_batch.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_dropout.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_batch_norm.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_25_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_100_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_150_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python cifar10_200_epochs.py")
    except Exception as e:
        pass


if __name__ == "__main__":
    func()
