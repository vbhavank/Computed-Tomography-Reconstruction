import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
from multiprocessing import Pool

from em import initialize, em, mle_em, mle_em_with_obj, mle_gd_with_obj


def run_gd(arg):
    np.random.seed()
    max_step, threshold, sparse, ix, norm = arg
    start = time.time()
    y_rand = np.random.poisson(Ax)
    # x_initial = np.random.randn(len(x_flat))
    alpha = 1.
    if norm > 1:
        alpha = 0.1
    print(f"process: {ix: 2d}, y[:10]: {y_rand[:10]}")
    x_et, diff, mse, objs, step = mle_gd_with_obj(max_step, A, y_rand, x_true=x_flat, threshold=threshold, x_initial=None, sparse=sparse, alpha=alpha)
    print(f"process: {ix: 2d} finished. step: {step: 2d}, mse: {mse: 8.2f}, diff: {diff: 8.2f} time consuming: {time.time() - start: 8.1f} seconds")
    return x_et, diff, mse, step

def run(arg):
    np.random.seed()
    max_step, threshold, sparse, ix = arg
    start = time.time()
    y_rand = np.random.poisson(Ax)
    alpha = 1
    # x_initial = np.random.randn(len(x_flat))
    print(f"process: {ix: 2d}, y[:10]: {y_rand[:10]}")
    x_et, diff, mse, objs, step = mle_em_with_obj(max_step, A, y_rand, x_true=x_flat, threshold=threshold, x_initial=None, sparse=sparse)
    print(f"process: {ix: 2d} finished. step: {step: 2d}, mse: {mse: 8.2f}, diff: {diff: 8.2f} time consuming: {time.time() - start: 8.1f} seconds")
    return x_et, diff, mse, step


def mc_mp(n, max_step=2000, threshold=10., sparse=True):
    pool = Pool()
    args = [(max_step, threshold, sparse, i) for i in range(n)]
    result = pool.map(run, args)
    pool.close()
    pool.join()
    return result

def mc_mp_gd(n, max_step=6000, threshold=10., sparse=True, norm=1):
    pool = Pool()
    args = [(max_step, threshold, sparse, i, norm) for i in range(n)]
    result = pool.map(run_gd, args)
    pool.close()
    pool.join()
    return result


def mc_run(n, max_step=500, threshold=10, sparse=True):
    etms = []
    for i in range(n):
        print(f"{i}th run start =>")
        y = np.random.poisson(Ax)
        x_et, diff, mse, step = mle_em(max_step, A, y, x_true=x_flat, threshold=threshold, x_initial=None, sparse=sparse)
        etms.append([x_et, diff, mse, step])
    return etms


def multi_mc(A_, x_):
    global A
    global Ax
    norms = np.logspace(-1, 1, 11)
    pows = np.linspace(-1, 1, 11)
    for norm, pow in zip(norms, pows):
        print(f"norm of A is: {norm: .4f}")
        A = A_original*norm
        Ax = A @ x_flat
        for repeat in range(200//n_para):
            try:
                etms = mc_mp(n=n_para, threshold=e_stop)
                etms_np = np.array(etms, dtype=object)
                np.save(f"{home}/etms_{time.time()}_batch_{n_para}_stop_{e_stop}_norm_10e{pow: .2f}.npy", etms_np)
            except Exception as e:
                print(f"except at {repeat}-th multiprocessing: {e}")
                
def multi_mc_gd(A_, x_):
    global A
    global Ax
    norms = np.logspace(-1, 1, 11)
    pows = np.linspace(-1, 1, 11)
    for norm, pow in zip(norms, pows):
        print(f"norm of A is: {norm: .4f}")
        A = A_original*norm
        Ax = A @ x_flat
        for repeat in range(epoch//n_para):
            try:
                etms = mc_mp_gd(n=n_para, threshold=e_stop, norm=norm)
                etms_np = np.array(etms, dtype=object)
                np.save(f"{home}/etms_{time.time()}_batch_{n_para}_stop_{e_stop}_norm_10e{pow: .2f}.npy", etms_np)
            except Exception as e:
                print(f"except at {repeat}-th multiprocessing: {e}")
                



if __name__ == "__main__":
    # A_original = sparse.load_npz("data/simulated_large_A_117_100.npz")
    # x_flat = np.load("data/simulated_large_x_117_100.npy")
    # A_original = sparse.load_npz("data/simulated_large_A_56_50.npz")
    # x_flat = np.load("data/simulated_large_x_56_50.npy")
    A_original = sparse.load_npz("data/simulated_large_A_23_10.npz")
    x_flat = np.load("data/simulated_large_x_23_10.npy")
    # y = np.load("constant_y/y.npy")
    # y = np.random.poisson(Ax)
    print("Image vector x is of shape {}".format(np.shape(x_flat)))
    print("Mixing matrix A is of shape {}".format(np.shape(A_original)))
    # print("Observation matrix Y is of shape {}".format(np.shape(y)))
    x_flat[0] = 20
    x_flat[90] =20
    
    epoch = 200
    n_para = 200
    e_stop = 0.0001
    home = "et10_em_rerun"
    # multi_mc(A_original, x_flat)
    multi_mc(A_original, x_flat)
    # multi_mc_gd(A_original, x_flat)
    # nm = 1
    # A = A_original*nm
    # Ax = A @ x_flat
    # run((10000, e_stop, True, 0))
    # run_gd((10000, e_stop, True, 0, nm))
    # etms = mc_run(2)
    # norm = 1
    # A = A_original*norm
    # Ax = A @ x_flat
    # for repeat in range(200//n_para):
    #     try:
    #         etms = mc_mp(n=n_para, threshold=e_stop)
    #         etms_np = np.array(etms, dtype=object)
    #         np.save(f"et50/etms_{time.time()}_batch_{n_para}_stop_{e_stop}_norm_{norm}.npy", etms_np)
    #         # np.save(f"constant_y/etms_{time.time()}_{n_para}_{e_stop}.npy", etms_np)
    #     except Exception as e:
    #         print(f"except at {repeat}-th multiprocessing: {e}")

