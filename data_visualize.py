import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(2)

overvations_filepath = "./observation_test_000.hdf5"
gt_filepath = "./ground_truth_test_000.hdf5"

# Because each file contains 128 slices
sample_ids = random.sample(range(0, 127), 10)

# Loading in the observations for the full 512 px image
# Note the ground truth is a center cropped version of the observation image
# Therefore one has to center crop the reconstructed image at 362px before evaluation
with h5py.File(overvations_filepath, "r") as f:
    for ids in sample_ids:
        sample_obervation_data = f['data'][ids, :, :]
print("Shape of observations for a single slice of scan before center crop is:{}".format(\
    np.shape(sample_obervation_data)))

# Loading ground truth file for the observation data loaded above
with h5py.File(gt_filepath, "r") as gtf:
    fig = plt.figure()
    for i, ids in enumerate(sample_ids):
        gt_image  = gtf['data'][ids, :, :]
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(gt_image, cmap='gray')
        ax.axis('off')
    fig.suptitle('Sample centre cropped(362px) test ground truth', fontsize=14)
print("Shape of ground truth for a single slice of scan is:{}".format(np.shape(gt_image)))
plt.show()

