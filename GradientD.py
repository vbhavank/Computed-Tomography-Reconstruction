import pdb

import scipy.io as sp
import numpy as np
import matplotlib.pyplot as plt


x = sp.loadmat("simulated_large.mat")
A_matrix = x['A']
y_observation = x['y'][0, :]

x_vec = x['x']

# Generating new x vector
g = np.random.randn(len(x_vec[0]))+ 5
u = np.random.uniform(0, 10, len(x_vec))
x = 0.2*g + u
x_vec = x.flatten().reshape(1, len(x_vec[0]))
Ax = A_matrix@x_vec.flatten()
y_observation = np.random.poisson(Ax).reshape(len(y_observation))

print("Image vector x is of shape {}".format(np.shape(x_vec)))
print("Mixing matrix A is of shape {}".format(np.shape(A_matrix)))
print("Obervation matrix Y is of shape {}".format(np.shape(y_observation)))

num_iteration = 100000
verbose_step = 2000

# Initialize x with vector of all ones.
x_init = np.ones(x_vec.shape)

class GradientDescent():
    def __init__(self,
                 y_observation,
                 A_matrix):
        self.y_obser = y_observation
        self.A_mat = A_matrix

    def compute_gradient(self):
        inner_term_final = 0
        Axn = (self.A_mat * self.x_old)
        for i in range(len(self.y_obser)):
            inner_term_final += (((self.y_obser[i] * self.A_mat[i, :]) / sum(Axn[i, :])) - self.A_mat[i, :])
        return inner_term_final

    def _get_new_x(self, x_old, lr_r=0.01):
        self.x_old = x_old
        gradient_step = self.compute_gradient()
        x_new = (x_old + lr_r*(gradient_step))
        return x_new

Gd_solver = GradientDescent(y_observation, A_matrix)
print("Initialization: {}".format(x_init))
for i in range(num_iteration):
    x_new = Gd_solver._get_new_x(x_init)
    diff = np.linalg.norm(x_new - x_init)
    mse = np.linalg.norm(x_new - x_vec)
    if i % verbose_step == 0:
        print(f'epoc: {i:2d}, diff: {diff: .4E}, mse: {mse:8.4f}')
    if (diff <= 1e-5):
        print(f'Convergence at: epoc: {i:2d}, diff: {diff: .4E}, mse: {mse:8.4f}')
        break
    x_init = x_new

print("Actual x_vector simulated:{}".format(x_vec))
print("Solution from Gradient Ascent:{}".format(x_new))
print("MSE of solution:{}".format(mse))

plt.figure()
plt.imshow(x_new.reshape(5, 5))
plt.axis('off')
plt.show()