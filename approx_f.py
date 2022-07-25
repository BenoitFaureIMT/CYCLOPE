import numpy as np
from scipy import stats

center = np.array([320, 320])
dists = np.array([[340, 340], [360, 360], [380, 380]])
corresponding_dists = np.array([3, 4, 5])
h = 5

dists = np.array([np.linalg.norm(d - center) for d in dists])
alphas = np.arctan(corresponding_dists/h)

f, inter, r_value, p_value, std_err = stats.linregress(np.sin(alphas), dists)
print(f, r_value)