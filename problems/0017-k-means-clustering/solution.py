import numpy as np

def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    if len(initial_centroids) != k:
        return []

    X = np.asarray(points, dtype=float)
    centroids = np.asarray(initial_centroids, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        return []

    for _ in range(max_iterations):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
		
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0: 
                new_centroids[j] = members.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return [tuple(c) for c in centroids.tolist()]