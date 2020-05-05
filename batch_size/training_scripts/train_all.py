import os
import time


def func():
    whole_start = time.time()
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
    whole_end = time.time()
    print('time cost for training all batch_size models is {} s'
          .format(round(whole_end - whole_start, 3)))


if __name__ == "__main__":
    func()
