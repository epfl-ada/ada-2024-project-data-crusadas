import numpy as np

# we have the following explained variance ratios from PCA:
explained_variance_ratios = np.array([0.02226398, 0.02605252, 0.01179182, 0.00950902,
                                      0.00892355, 0.00662831, 0.0061859, 0.00530022])

# Cluster means from the analysis (PC1 through PC8)
cluster_means = {
    0: [0.249053, -0.080032, -0.017682, -0.069967,  0.038991,  0.061582,  0.005178,  0.000579],
    1: [0.262984,  0.189522, -0.085679,  0.116146, -0.155535,  0.174491, -0.148607,  0.021532],
    2: [0.379506, -0.149562,  0.128452,  0.065387, -0.018421,  0.000803,  0.000122, -0.005868],
    3: [0.292106,  0.269329,  0.050015,  0.011941,  0.072518, -0.003838,  0.010615, -0.002157],
    4: [0.263478, -0.082743, -0.103096,  0.029240,  0.074446, -0.061853, -0.069044, -0.006481],
    5: [0.286028,  0.037230, -0.024730, -0.064091, -0.082715, -0.047946,  0.007292,  0.005726],
    6: [0.162790, -0.032305,  0.010795,  0.005012,  0.005087,  0.005957,  0.000144,  0.014120],
    7: [0.227468, -0.026946, -0.146283,  0.101795, -0.000715,  0.010981,  0.133357,  0.003663],
    8: [0.237599, -0.054014, -0.056930, -0.032819,  0.018249,  0.008775, -0.022155,  0.006228]
}

clusters = np.array(list(cluster_means.keys()))
mean_vectors = np.array(list(cluster_means.values()))

# Compute mean and std across clusters to standardize data.
pc_mean = mean_vectors.mean(axis=0)
pc_std = mean_vectors.std(axis=0, ddof=1)
# Ensure no division by zero
pc_std[pc_std == 0] = 1.0

# Compute weights for each dimension from explained variance.
dimension_weights = np.sqrt(explained_variance_ratios)

# Example user answers (from UI sliders, -2 to 2)
user_answers = {
    "PC1": 1.5,
    "PC2": -1.0,
    "PC3": 2.0,
    "PC4": 0.0,
    "PC5": -1.5,
    "PC6": 1.0,
    "PC7": -0.5,
    "PC8": 2.0,
}

# Validate user input
for k, v in user_answers.items():
    if not (isinstance(v, (int, float)) and -2 <= v <= 2):
        raise ValueError(f"Slider value for {k} must be a number between -2 and 2.")

# Convert user answers to array
user_vector = np.array(
    [
        user_answers["PC1"],
        user_answers["PC2"],
        user_answers["PC3"],
        user_answers["PC4"],
        user_answers["PC5"],
        user_answers["PC6"],
        user_answers["PC7"],
        user_answers["PC8"],
    ]
)

# Standardize cluster means and user vector using pc_mean and pc_std
standardized_mean_vectors = (mean_vectors - pc_mean) / pc_std
standardized_user_vector = (
    user_vector  # interpreted as std dev offsets from mean
)

# Weighted space transformation
weighted_mean_vectors = standardized_mean_vectors * dimension_weights
weighted_user_vector = standardized_user_vector * dimension_weights

# Euclidean distances in the weighted space
dists = np.linalg.norm(weighted_mean_vectors - weighted_user_vector, axis=1)

closest_cluster = clusters[np.argmin(dists)]

print("User's answers (raw):", user_vector)
# print("Cluster means (raw):", mean_vectors)
print("Mean of PCs across clusters:", pc_mean)
print("Std of PCs across clusters:", pc_std)
# print("Standardized cluster means:", standardized_mean_vectors)
print("User vector (interpreted as std units):", standardized_user_vector)
print("Weights (sqrt of explained variance ratios):", dimension_weights)
print("Distances to each cluster (weighted standardized):", dict(zip(clusters, dists)))
print("The user is closest to cluster:", closest_cluster)