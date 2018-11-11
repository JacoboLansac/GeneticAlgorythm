import time


def timer(func_to_time):
    """
    """
    def decorator(*args, **kwargs):
        t0 = time.time()
        print("Running {}()...".format(func_to_time.__name__), end=' ')
        results = func_to_time(*args, **kwargs)
        print("{} secs".format(round(time.time() - t0, 2)))
        return results

    return decorator

