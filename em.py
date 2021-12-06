import numpy as np

def initialize(A, y):
    N, M = A.shape
    x_start = np.zeros(M)
    for k, o in enumerate(y):
        aver = o/np.sum(A[k])
        x_start += A[k]*aver
        # print(k, o, aver)
    x_start = x_start/N
    return np.array(x_start).flatten()

def em(A, y, x_old):
    N, M = A.shape
    z = np.zeros((N, M))
    for i in range(N):
        z[i] = y[i]*A[i]*x_old/np.sum(A[i]*x_old)
    x_new = np.zeros(M)
    for j in range(M):
        x_new[j] = np.sum(z[:, j])/(np.sum(A[:, j]))
    return x_new

def em_bdct(A, y, x_old, sparse=False):
    N, M = A.shape
    if sparse:
        A_dense = np.array(A.todense())
    else:
        A_dense = A.copy()
    m = A_dense*x_old
    v = y/A.dot(x_old)
    z = (m.T*v).T
    x_new = np.array(np.sum(z, axis=0)/np.sum(A, axis=0)).flatten()
    return x_new

def gradient(A, y, x_old, sparse=False, lr=0.001):
    g = y/A.dot(x_old)
    gradient = (A.T*g).T - np.array(A.sum(axis=0)).flatten()
    x_new = x_old + lr*gradient
    return x_new

def mle_em(max_iter, A, y, x_true, threshold=1, x_initial=None, sparse=False):
    if x_initial is None:
        x_old = initialize(A, y)
    else:
        x_old = x_initial
    mse = []
    for i in range(max_iter):
        x_new = em_bdct(A, y, x_old, sparse)
        mse = np.linalg.norm(x_new-x_true)
        diff = np.linalg.norm(x_new-x_old)
        if i%20 == 0:
            print(f'step: {i}, diff: {diff}, mse: {mse}')
        if diff < threshold:
            return x_new, diff, mse, i
        x_old = x_new
    return x_new, diff, mse, max_iter


def mle_em_with_obj(max_iter, A, y, x_true, threshold=1, x_initial=None, sparse=False):
    if x_initial is None:
        x_old = initialize(A, y)
    else:
        x_old = x_initial
    mse = []
    objs = []
    for i in range(max_iter):
        x_new = em_bdct(A, y, x_old, sparse)
        mse = np.linalg.norm(x_new-x_true)
        Ax_new = A.dot(x_new)
        obj = -np.log(Ax_new)
        obj = (obj*y + Ax_new).sum()
        if len(objs) > 1:
            diff = objs[-1] - obj
        else:
            diff = 100   
        objs.append(obj)
        if i%20 == 0:
            print(f'step: {i}, diff: {diff}, mse: {mse}:, obj: {obj}')
        if diff < threshold:
            return x_new, diff, mse, objs, i
        x_old = x_new
        objs.append(obj)
        # diff = np.linalg.norm(x_new-x_old)
        # if i%20 == 0:
        #     print(f'step: {i}, diff: {diff}, mse: {mse}:, obj: {obj}')
        # if diff < threshold:
        #     return x_new, diff, mse, objs, i
        # x_old = x_new
    return x_new, diff, mse, objs, max_iter


def mle_gd_with_obj(max_iter, A, y, x_true, threshold=1, x_initial=None, sparse=False, alpha=0.001):
    if x_initial is None:
        x_old = initialize(A, y)
    else:
        x_old = x_initial
    mse = []
    objs = []
    lr = alpha
    for i in range(max_iter):
        # lr = alpha
        early_stop = False
        x_new = gradient(A, y, x_old, sparse, lr=lr)
        Ax_new = A.dot(x_new)
        obj = -np.log(Ax_new)
        obj = (obj*y + Ax_new).sum()
        # line_search = 10
        # if len(objs)>0 and obj >= objs[-1]:
        #     lr *= 0.5
        #     x_new = x_old 
        #     Ax_new = A.dot(x_new)
        #     obj = -np.log(Ax_new)
        #     obj = (obj*y + Ax_new).sum()
            # objs.append(objs[-1])
        mse = np.linalg.norm(x_new-x_true)
        if len(objs) > 1:
            diff = objs[-1] - obj
        else:
            diff = 100   
        objs.append(obj)
        if i%20 == 0:
            print(f'step: {i}, lr: "{lr: .6e}" diff: {diff}, mse: {mse}:, obj: {obj}')
        if diff < threshold:
            return x_new, diff, mse, objs, i
        x_old = x_new
    return x_new, diff, mse, objs, max_iter