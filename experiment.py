import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
from multiprocessing import Pool

from em import initialize, em, mle_em


def run(arg):
    np.random.seed()
    max_step, threshold, sparse, ix = arg
    start = time.time()
    y_rand = np.random.poisson(Ax)
    # x_initial = np.random.randn(len(x_flat))
    print(f"process: {ix: 2d}, y[:10]: {y_rand[:10]}")
    x_et, diff, mse, step = mle_em(max_step, A, y_rand, x_true=x_flat, threshold=threshold, x_initial=None, sparse=sparse)
    print(f"process: {ix: 2d} finished. step: {step: 2d}, mse: {mse: 8.2f}, diff: {diff: 8.2f} time consuming: {time.time() - start: 8.1f} seconds")
    return x_et, diff, mse, step


def mc_mp(n, max_step=500, threshold=10, sparse=True):
    pool = Pool()
    args = [(max_step, threshold, sparse, i) for i in range(n)]
    result = pool.map(run, args)
    return result


def mc_run(n, max_step=500, threshold=10, sparse=True):
    etms = []
    for i in range(n):
        print(f"{i}th run start =>")
        y = np.random.poisson(Ax)
        x_et, diff, mse, step = mle_em(max_step, A, y, x_true=x_flat, threshold=threshold, x_initial=None, sparse=sparse)
        etms.append([x_et, diff, mse, step])
    return etms


if __name__ == "__main__":
    A_original = sparse.load_npz("data/simulated_large_A_117_100.npz")
    x_flat = np.load("data/simulated_large_x_117_100.npy")
    y = np.load("constant_y/y.npy")
    # y = np.random.poisson(Ax)
    print("Image vector x is of shape {}".format(np.shape(x_flat)))
    print("Mixing matrix A is of shape {}".format(np.shape(A_original)))
    print("Observation matrix Y is of shape {}".format(np.shape(y)))


    n_para = 20
    e_stop = 10
    # etms = mc_run(2)
    A = A_original
    Ax = A @ x_flat
    for repeat in range(200//n_para):
        try:
            etms = mc_mp(n=n_para, threshold=e_stop)
            etms_np = np.array(etms, dtype=object)
            np.save(f"estimations/etms_{time.time()}_{n_para}_{e_stop}.npy", etms_np)
            # np.save(f"constant_y/etms_{time.time()}_{n_para}_{e_stop}.npy", etms_np)
        except Exception as e:
            print(f"except at {repeat}-th multiprocessing: {e}")

    norm = 0.1
    for norm in [0.1, 0.5, 5, 10]:
        A = A_original*norm
        Ax = A @ x_flat
        for repeat in range(200//n_para):
            try:
                etms = mc_mp(n=n_para, threshold=e_stop)
                etms_np = np.array(etms, dtype=object)
                np.save(f"estimations/etms_{time.time()}_{n_para}_{e_stop}_{norm}.npy", etms_np)
                # np.save(f"constant_y/etms_{time.time()}_{n_para}_{e_stop}.npy", etms_np)
            except Exception as e:
                print(f"except at {repeat}-th multiprocessing: {e}")