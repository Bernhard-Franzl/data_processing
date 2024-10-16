from forcasting._preprocessing.preprocessing import PLCount
import numpy as np

data_test = np.array([
    [0, 1.00, 0],
    [1, 1.00, 1],
    [0, 1.00, 1],
    [1, 1.00, 2],
    [0, 1.00, 2],
    [0, 1.00, 2],
    [2, 1.41, 4],
    [0, 1.00, 4],
    [3, 1.73, 7],
    [1, 1.00, 8],
    [-2, 1.41, 6],
    [-7, 2.65, -1],
    [0, 1.00, -1]
])

n = data_test.shape[0]
cc_max = int(data_test[:,2].max())
#m = int(cc_max + (cc_max * 0.2))
m = cc_max

plcount = PLCount()
sigma_array = data_test[:,1]
delta_array = data_test[:,0]
plcount.run_algorithm(
    n=n,
    m=m,
    sigma_array=sigma_array,
    delta_array=delta_array
)

plcount.run_algorithm_vectorized(
    n=n,
    m=m,
    sigma_array=sigma_array,
    delta_array=delta_array
)

