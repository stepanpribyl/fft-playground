from functools import wraps
import numpy as np

# Module-level history of the sum of squared residuals, one entry appended per
# call of _varpro_resid (i.e. per optimiser function evaluation). Useful for
# monitoring convergence; clear it with ``results.clear()`` before a new fit.
residuals = []
params = []


def register_results():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # print(*args)
            global params, residuals
            result = func(*args, **kwargs)
            residuals.append(sum([r**2 for r in result]))
            if not len(params):
                # print("INITIATING PARAMS")
                params = [[] for _ in range(len(kwargs))]

            for i, key in enumerate(kwargs.keys()):
                # print("FILLING PARAMS")
                params[i].append(kwargs[key])

            return result
        return wrapper

    return decorator

def apply_sliding_filter(data, sliding_filter):
    data_filtered = data.copy()
    sliding_filter_normed = sliding_filter / np.sum(sliding_filter)
    for i in range(len(sliding_filter) - 1, len(data)):
        data_filtered[i] = sum(
            [data[i - j] * sliding_filter_normed[-j - 1] for j in range(len(sliding_filter_normed))])
    return data_filtered