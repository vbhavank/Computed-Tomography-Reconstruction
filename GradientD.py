import scipy.io as sp
import numpy as np
import matplotlib.pyplot as plt


x = sp.loadmat("simulated_heart.mat")
A_matrix = x['A']
x_vec = x['x']
y_observation = x['y']
num_iteration = 1000
verbose_step = 200

x_init_value = y_observation[0, 0]/sum(A_matrix[0, :])
x_location = np.where(A_matrix[0, :] == 1)

x_init = np.ones(x_vec.shape)

for x_lo in x_location:
    x_init[0, x_lo] = x_init_value

class GradientDescent():
    def __init__(self,
                 y_observation,
                 A_matrix):
        self.y_obser = y_observation
        self.A_mat = A_matrix

    def compute_gradient(self):
        inner_term_final = 0
        for i in range(len(self.y_obser[0, :])):
            inner_term1 = ((self.y_obser[0, i] * self.A_mat[i, :]) / ((self.A_mat * self.x_old)[i, :]))
            # Warning: Setting nan values to zero
            inner_term1[np.isnan(inner_term1)] = 0
            inner_term_final += (inner_term1 - self.A_mat[i, :])
        return inner_term_final

    def _get_new_x(self, x_old, lr_r=0.0001):
        self.x_old = x_old
        x_new = np.zeros(x_old.shape)
        gradient_step = self.compute_gradient()
        for i, x_j in enumerate(x_old[0, :]):
            x_new[0, i] = (x_j + lr_r*(gradient_step))[i]
        return x_new

Gd_solver = GradientDescent(y_observation, A_matrix)
print("Initialization: {}".format(x_init))
for i in range(num_iteration):
    x_new = Gd_solver._get_new_x(x_init)
    x_init = x_new
    if i % verbose_step == 0:
        print("After {} iteration: {}".format(i, x_new))
print("Actual x_vector simulated:{}".format(x_vec))
mse = sum(((x_new - x_vec)**2)[0])
print("MSE of solution:{}".format(mse))

plt.figure()
plt.imshow(x_new.reshape(5, 5))
plt.axis('off')
plt.show()