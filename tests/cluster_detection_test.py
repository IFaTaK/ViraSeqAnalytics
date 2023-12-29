import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.cluster_detection import KMeansClustering

class TestKMeansClustering(unittest.TestCase):

    @staticmethod
    def generate_three_cluster_data(num_samples=100, spread=0.5):
        centers = np.array([[1, 1], [5, 5], [8, 1]])
        data = []
        for center in centers:
            data.append(center + np.random.randn(num_samples, 2) * spread)
        return np.vstack(data)

    @classmethod
    def setUpClass(cls):
        # Synthetic dataset for testing
        cls.data = cls.generate_three_cluster_data()

    def test_kmeans_clustering(self):
        data = self.generate_three_cluster_data()
        kmeans = KMeansClustering(data, k=3)
        self.assertEqual(kmeans.k, 3)

    def test_centroid_initialization(self):
        kmeans = KMeansClustering(self.data, k=3)
        self.assertEqual(kmeans.centroids.shape, (3, 2))

    def test_cluster_assignments(self):
        kmeans = KMeansClustering(self.data, k=3)
        unique_labels = np.unique(kmeans.labels)
        self.assertEqual(len(unique_labels), 3)

    def test_silhouette_scores(self):
        kmeans = KMeansClustering(self.data, k=3)
        silhouette_scores = kmeans._silhouette_scores()
        self.assertEqual(len(silhouette_scores), len(self.data))

    def test_optimize_k(self):
        kmeans = KMeansClustering(self.data)
        optimal_k = kmeans.optimize_k(max_cluster=5)
        self.assertEqual(optimal_k, 3)

    def test_multiple_initializations(self):
        kmeans = KMeansClustering(self.data, k=3, n_init=5)
        self.assertIsNotNone(kmeans.centroids)

    @classmethod
    def tearDownClass(cls):
        # Plot the results
        data = cls.generate_three_cluster_data()
        kmeans = KMeansClustering(data, k=3)
        cls.plot_clusters(data, kmeans.labels, kmeans.centroids)

    @staticmethod
    def plot_clusters(data, labels, centroids):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
        plt.title("KMeans Clustering Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

if __name__ == '__main__':
    unittest.main()
 