import os


def func():
    try:
        os.system("python dropout_25_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_100_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_150_epochs.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_200_epochs.py")
    except Exception as e:
        pass


if __name__ == "__main__":
    func()
