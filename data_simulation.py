import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


# Hardcoded binary image for illustration of a heart
sample_x = np.array([[1, 0, 1, 0, 1],
		[0, 1, 0, 1, 0],
		[0, 1, 1, 1, 0],
		[1, 0, 1, 0, 1],
		[1, 1, 0, 1, 1]])

x, y = sample_x.shape
string_list = []
for i in range(x):
	for j in range(y):
		string_list.append("{},{}".format(i, j))
indices_A = np.array(string_list, dtype = 'object').reshape(x, y)

# Random alpha scaling to prevent trivial 1, 0 problem
# import random
# alpha_scale = random.sample(range(1, 8), y)
# for i, alpha in enumerate(alpha_scale):
# 	sample_x[:, i] = alpha * sample_x[:, i]

# Alternatively hardcoded intensity scaling for reproducibility
sample_x[:, 1] =  5*sample_x[:, 1]
sample_x[:, 2] =  2*sample_x[:, 2]
sample_x[:, 3] =  3*sample_x[:, 3]
sample_x[:, 4] =  6*sample_x[:, 4]
sample_x[:, 0] =  4*sample_x[:, 0]

# Obtain actual vector x used in model
x_flatten = sample_x.flatten()

# Visualize the image
print(sample_x)
plt.figure()
plt.imshow(sample_x)
plt.title('Sample Image')
plt.show()
plt.close()

# Computing rays along all possible diagonals and their respective indices ('row', 'column')
def sample_diag(sample_x):
	diags = [sample_x[::-1, :].diagonal(i) for i in range(-sample_x.shape[0]+1, sample_x.shape[1])]
	diags.extend(sample_x.diagonal(i) for i in range(sample_x.shape[1]-1, -sample_x.shape[0], -1))
	return [n.tolist() for n in diags]
all_diags = sample_diag(sample_x)
diag_indic = sample_diag(indices_A)

# Computing rays along all columns
column_arr = []
column_indic_global = []
for j in range(y):
	columns = [row[j] for row in sample_x]
	column_arr.append(columns)
	column_indic = []
	for r_cnt in range(x):
		column_indic.append("{},{}".format(r_cnt, j))
	column_indic_global.append(column_indic)

# Computing rays along all rows
rows_arr = []
row_indic_global = []
for j in range(x):
	rows = [row for row in sample_x[j, :]]
	rows_arr.append(rows)
	row_indic = []
	for c_cnt in range(y):
		row_indic.append("{},{}".format(j, c_cnt))
	row_indic_global.append(row_indic)

# Joining diagonals, columns and row indices and values into a single list
all_indices = [*diag_indic, *column_indic_global, *row_indic_global]
all_values_combo = [*all_diags, *column_arr, *rows_arr]

# Creating empty mixing matrix A for model
row_dim, col_dim = np.shape(sample_x)
A_mixing_matrix = np.zeros((len(all_indices), row_dim, col_dim))
for obs_n, ind in enumerate(all_indices):
	for sub_i in ind:
		local_x, local_y = sub_i.split(',')
		A_mixing_matrix[obs_n, int(local_x), int(local_y)] = 1
A_final = A_mixing_matrix.reshape((obs_n+1, row_dim*col_dim))

# Pure observations are used for lambda to sample from poisson distribution
pure_obser = np.sum(A_final*x_flatten, axis=1)
poisson_obser = np.random.poisson(pure_obser, len(all_indices))

# Saving to mat files for use in MATLAB
export_dict = {"A": A_final, "y":poisson_obser, "x": x_flatten}

savemat("simulated_heart.mat", export_dict)
print("Image vector x is of shape {}".format(len(x_flatten)))
print("Mixing matrix A is of shape {}".format(np.shape(A_final)))
print("Obervation matrix Y is of shape {}".format(np.shape(poisson_obser)))
print("X:", x_flatten)
print("A:", A_final)
print("Y:", poisson_obser)
print("Y without poisson", pure_obser)

