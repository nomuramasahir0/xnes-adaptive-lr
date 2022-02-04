import numpy as np


def sphere(x):
    return np.sum(x ** 2)


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    Dell = np.diag([10 ** (3 * i / (n - 1)) for i in range(n)])
    return sphere(Dell @ x)


def rastrigin(x):
    n = len(x)
    if n < 2:
        raise ValueError("dimension must be greater one")
    return 10 * n + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def bohachevsky(x):
    n = len(x)
    f_value = 0.0
    for i in range(n - 1):
        f_value += pow(x[i], 2.0)
        f_value += 2 * pow(x[i + 1], 2.0)
        f_value -= 0.3 * np.cos(3 * np.pi * x[i])
        f_value -= 0.4 * np.cos(4 * np.pi * x[i + 1])
        f_value += 0.7

    return f_value


def get_problem(problem):
    if problem == "sphere":
        obj_func = sphere
    elif problem == "ellipsoid":
        obj_func = ellipsoid
    elif problem == "rastrigin":
        obj_func = rastrigin
    elif problem == "bohachevsky":
        obj_func = bohachevsky
    else:
        raise NotImplementedError

    return obj_func
