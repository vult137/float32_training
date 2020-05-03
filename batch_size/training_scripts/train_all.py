import os


def func():
    try:
        os.system("python dropout_32_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_64_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_128_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_256_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python dropout_512_batch_size.py")
    except Exception as e:
        pass

    try:
        os.system("python regular_32_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python regular_64_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python regular_128_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python regular_256_batch_size.py")
    except Exception as e:
        pass
    try:
        os.system("python regular_512_batch_size.py")
    except Exception as e:
        pass


if __name__ == "__main__":
    func()
