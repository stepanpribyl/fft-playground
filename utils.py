from functools import wraps

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