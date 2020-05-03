import os


def func():
    try:
        os.system("python cifar10_regular_2.py")
    except Exception as e:
        pass

    try:
        os.system("python cifar10_regular_4.py")
    except Exception as e:
        pass

    try:
        os.system("python cifar10_dropout_2.py")
    except Exception as e:
        pass

    try:
        os.system("python cifar10_dropout_4.py")
    except Exception as e:
        pass


if __name__ == "__main__":
    func()
