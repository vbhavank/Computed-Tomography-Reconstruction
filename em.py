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