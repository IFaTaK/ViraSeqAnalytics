"""
This module implements the K-means clustering algorithm from scratch. 
It includes a class for performing K-means clustering on a dataset, along with 
methods for initializing centroids, iterating to find cluster assignments, and 
determining the optimal number of clusters.

The K-means algorithm is a widely used method for unsupervised machine learning, 
particularly useful in identifying distinct groups or clusters within a dataset.

Classes:
    KMeansClustering: A class that implements the K-means clustering algorithm.
"""
import numpy as np


class KMeansClustering:
    """
    A class that implements the K-means clustering algorithm.

    This class performs clustering on a given dataset, allowing the user to specify
    the number of clusters (k) and the number of centroid initializations (n_init).
    It provides functionality to compute silhouette scores and determine the 
    optimal number of clusters.

    Attributes:
        data (np.ndarray): The dataset to be clustered.
        k (int): The number of clusters.
        n_init (int): The number of times the algorithm will be run with different centroid seeds.
        labels (np.ndarray): The cluster assignment for each data point.
        centroids (np.ndarray): The coordinates of the cluster centroids.

    Methods:
        fit(max_iterations): Performs K-means clustering on the data.
        _silhouette_scores(): Calculates silhouette scores for the current clustering.
        optimize_k(max_cluster): Determines the optimal number of clusters based on silhouette scores.
    """

    def __init__(self, data, k=3, n_init=10):
        """
        Initializes the KMeansClustering class with the dataset, number of clusters, and initializations.

        Args:
            data (np.ndarray): The dataset to be clustered.
            k (int): The number of clusters.
            n_init (int): The number of times the algorithm will be run with different centroid seeds.
        """
        self.data = data
        self.k = k
        self.n_init = n_init
        self.labels = None
        self.centroids = None
        self.fit()

    @staticmethod
    def euclidean_distance(data_point, centroids):
        """
        Calculates the Euclidean distance between a data point and each centroid.

        Args:
            data_point (np.ndarray): A single data point.
            centroids (np.ndarray): Array of centroids.

        Returns:
            np.ndarray: Euclidean distances from the data point to each centroid.
        """
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def _initialize_centroids(self):
        """
        Initializes centroids randomly within the range of the dataset.

        Returns:
            np.ndarray: Randomly initialized centroids.
        """
        return np.random.uniform(np.amin(self.data, axis=0), np.amax(self.data, axis=0),
                                 size=(self.k, self.data.shape[1]))

    def _kmeans_run(self, centroids, max_iterations):
        """
        Performs a single run of the K-means algorithm given initial centroids.

        Args:
            centroids (np.ndarray): Initial centroids for this run.
            max_iterations (int): Maximum number of iterations for convergence.

        Returns:
            tuple: Final centroids and cluster labels after convergence.
        """
        for _ in range(max_iterations):
            y = []
            for data_point in self.data:
                distances = KMeansClustering.euclidean_distance(data_point, centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)
            new_centroids = np.array([self.data[y == i].mean(axis=0) if len(self.data[y == i]) > 0 else centroids[i]
                                      for i in range(self.k)])

            if np.allclose(centroids, new_centroids, atol=0.001):
                break
            else:
                centroids = new_centroids

        return centroids, y

    def fit(self, max_iterations=200):
        """
        Applies the K-means clustering algorithm to the dataset.

        This method iteratively refines the positions of centroids and reassigns data points
        to clusters until convergence or maximum iterations are reached.

        Args:
            max_iterations (int): Maximum number of iterations for the algorithm to run.
        """
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            initial_centroids = self._initialize_centroids()
            centroids, labels = self._kmeans_run(initial_centroids, max_iterations)

            inertia = np.sum([np.min(self.euclidean_distance(data_point, centroids)**2) for data_point in self.data])
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels

    def _silhouette_scores(self):
        """
        Calculate the silhouette score for the current clustering.

        Returns:
            np.ndarray: The silhouette score for each sample.
        """
        data = self.data
        labels = self.labels

        def euclidean_distance(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))

        silhouette_scores = np.zeros(data.shape[0])

        for idx, point in enumerate(data):
            cluster_label = labels[idx]

            # Intra-cluster distance
            same_cluster = data[labels == cluster_label]
            distances = [euclidean_distance(point, other_point) for other_point in same_cluster if not np.array_equal(point, other_point)]
            if len(distances)==0:
                silhouette_scores[idx] = 0
                continue
            a = np.mean(distances)

            # Nearest-cluster distance
            b = np.inf
            for other_label in np.unique(labels):
                if other_label != cluster_label:
                    other_cluster = data[labels == other_label]
                    if len(other_cluster) == 0:
                        other_distance = np.inf
                    else:
                        other_distance = np.mean([euclidean_distance(point, other_point) for other_point in other_cluster])
                    b = min(b, other_distance)

            silhouette_scores[idx] = (b - a) / max(a, b)

        return silhouette_scores

    def optimize_k(self, max_cluster=7):
        """
        Finds the optimal number of clusters (k) for KMeans clustering.

        This method uses the silhouette score to determine the optimal k. It calculates
        the mean silhouette scores for different values of k and returns the k value
        that appears optimal based on the highest silhouette score.

        Args:
            max_cluster (int): The maximum number of clusters to consider.

        Returns:
            int: Optimal number of clusters based on the silhouette score.
        """

        mean_silhouette_scores = []
        range_k = range(2, max_cluster + 1)

        for k in range_k:
            kmeans_temp = KMeansClustering(self.data, k)
            silhouette_scores = kmeans_temp._silhouette_scores()
            mean_silhouette_scores.append(np.mean(silhouette_scores))

        # Finding the optimal k using the silhouette score
        optimal_k_silhouette = np.argmax(mean_silhouette_scores) + 2  # +2 since range starts from 2
        return optimal_k_silhouette
