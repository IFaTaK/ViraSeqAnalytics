import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.pca import PCAAnalysis

def generate_test_data(n_samples=1000, variance=(10, 3, 2, 1), noise_level=0.1, random_seed=42):
    """
    Generate test data with specified variance along principal axes.

    Args:
        n_samples (int): Number of data points to generate.
        variance (tuple): Variances along the principal axes.
        noise_level (float): Standard deviation of Gaussian noise added to data.
        random_seed (int): Seed for random number generator for reproducibility.

    Returns:
        np.ndarray: Generated test data.
    """
    np.random.seed(random_seed)
    # Generate data with specified variance
    X = np.random.randn(n_samples, 4) * np.sqrt(variance)

    # Optionally apply a linear transformation (rotation here)
    theta = np.pi / 4  # 45 degrees rotation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta),0,0],
                                [np.sin(theta),  np.cos(theta),0,0],
                                [0,0,1,0],
                                [0,0,0,1]])
    X = np.dot(X, rotation_matrix)

    # Add Gaussian noise
    noise = noise_level * np.random.randn(n_samples, 4)
    X += noise

    return X

class TestPCAAnalysis(unittest.TestCase):

    def setUp(self):
        # Reset PCAAnalysis instance before each test
        X = generate_test_data()
        self.pca = PCAAnalysis(X, n_components=2)

    def test_fit(self):
        # Test that PCA fitting works and principal components are computed
        self.assertEqual(self.pca.components.shape, (4, 2))

    def test_explained_variance_ratio(self):
        # Test that the explained variance ratio is calculated
        total_variance = sum(self.pca.explained_variance_ratio)
        self.assertTrue(0 <= total_variance <= 1)

    def test_transformed_data_2d(self):
        # Test the 2D transformation
        transformed_data = self.pca.get_transformed_data_2d()
        self.assertEqual(transformed_data.shape, (1000, 2))

    def test_transformed_data_3d_exception(self):
        # Test that 3D transformation raises an exception with less than 3 components
        with self.assertRaises(ValueError):
            self.pca.get_transformed_data_3d()

    def test_plot_2d(self):
        # Test 2D plotting (visual check)
        self.pca.plot_2d()
        # plt.show()
        # plt.close()

    def test_plot_3d_exception(self):
        # Test that 3D plotting raises an exception with less than 3 components
        with self.assertRaises(ValueError):
            self.pca.plot_3d()
            
    def test_plot_3d(self):
        # Test that 3D plotting raises an exception with less than 3 components
        self.pca.set_num_components(3)
        self.pca.plot_3d()
        # plt.show()
        # plt.close()
        
    def test_biplot_2d(self):
        labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
        self.pca.biplot_2d(labels=labels)
        # plt.show()
        # plt.close()
        
    def test_biplot_3d(self):
        labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
        self.pca.set_num_components(3)
        self.pca.biplot_3d(labels=labels)
        # plt.show()
        # plt.close()

    def test_optimal_num_components(self):
        # Test the optimal number of components function
        optimal_components = self.pca.optimal_num_components()
        self.assertEqual(optimal_components, 4)

    def test_plot_correlation_circle(self):
        # Test scree plot (visual check)
        labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
        self.pca.plot_correlation_circle(labels=labels)
        # plt.show()
        # plt.close()

    def test_plot_scree(self):
        # Test scree plot (visual check)
        self.pca.plot_scree()
        # plt.show()
        # plt.close()
    
    def test_plot_distance_from_origin(self):
        # Test distance from origin plot (visual check)
        self.pca.plot_distance_from_origin()
        # plt.show()
        # plt.close()
        
    def test_plot_explained_variance(self):
        # Test explained variance plot (visual check)
        self.pca.plot_explained_variance()
        # plt.show()
        # plt.close()

    def test_set_num_components(self):
        # Test setting a new number of components
        self.pca.set_num_components(3)
        self.assertEqual(self.pca.n_components, 3)

    def test_set_data(self):
        # Test setting new data
        new_data = np.random.randn(100, 4)
        self.pca.set_data(new_data)
        self.assertEqual(self.pca.data.shape, new_data.shape)


    @classmethod
    def tearDownClass(cls):
        # Code here will run once after all tests
        plt.show()


# Run the tests
if __name__ == '__main__':
    unittest.main()
