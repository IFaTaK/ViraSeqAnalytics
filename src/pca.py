"""
This module implements Principal Component Analysis (PCA) from scratch, providing tools for 
dimensionality reduction and data visualization in 2D and 3D. The PCAAnalysis class within 
allows for a detailed exploration of high-dimensional data by projecting it onto principal 
components, thereby uncovering patterns and relationships in the data. Additionally, the 
module includes functions to plot various informative charts such as scree plots, 
correlation circles, and 3D scatter plots.

Key Features:
- Implementation of PCA without relying on external libraries like sklearn.
- Functions to plot data in 2D and 3D PCA transformed space.
- Calculation of explained variance and correlation metrics.
- Tools for optimal component selection and data transformation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class PCAAnalysis:
    """
    A class for performing Principal Component Analysis on a given dataset.

    This class calculates principal components of the data and provides methods 
    for data transformation, visualization, and analysis. It includes functionalities 
    to plot the data in reduced dimensions, assess the quality of representation, 
    and understand the contribution of original variables to principal components.

    Attributes:
        data (np.ndarray): The original high-dimensional data.
        n_components (int): Number of principal components to retain.
        mean (np.ndarray): Mean of the data, used for centering.
        components (np.ndarray): Principal components (eigenvectors).
        transformed_data (np.ndarray): Data projected onto principal components.
        explained_variance_ratio (np.ndarray): Ratio of variance explained by each principal component.
        correlations (np.ndarray): Correlations between original variables and components.

    Methods:
        set_num_components(n_components): Updates the number of components and refits the model.
        set_data(data): Updates the data and refits the model.
        optimal_num_components(threshold): Determines optimal number of components for given variance threshold.
        get_transformed_data_2d(): Returns 2D transformed data.
        get_transformed_data_3d(): Returns 3D transformed data.
        plot_2d(point_size): Plots 2D transformed data with quality representation.
        plot_3d(point_size): Plots 3D transformed data with quality representation.
        plot_correlation_circle(): Plots correlation between original variables and principal components.
        plot_distance_from_origin(): Plots distance of each sample from origin in PCA space.
        plot_explained_variance(): Plots explained variance by each principal component.
        plot_scree(): Plots cumulative explained variance by principal components (Scree plot).
    """
    def __init__(self, data, n_components=2):
        self.data = data
        self.n_components = n_components if n_components else data.shape[1]
        self._fit()

    def _fit(self):
        # Center the data
        self.mean = np.mean(self.data, axis=0)
        self.data_centered = self.data - self.mean

        # Calculate covariance matrix and eigen decomposition
        covariance_matrix = np.cov(self.data_centered.T)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:,idx]

        # Compute total variance and explained variance ratio
        total_variance = np.sum(eigen_values)
        self.explained_variance_ratio = eigen_values[:self.n_components] / total_variance

        # Project the data onto the principal components
        self.components = eigen_vectors[:, :self.n_components]
        self.transformed_data = self.data_centered.dot(self.components)

        # Compute correlations between original variables and components
        self.correlations = np.array([np.corrcoef(self.data_centered[:, i], self.transformed_data[:, j])[0, 1]
                                      for i in range(self.data_centered.shape[1])
                                      for j in range(self.n_components)]).reshape(self.data_centered.shape[1], self.n_components)

    def set_num_components(self, n_components):
        """
        Sets the number of principal components and refits the PCA model.

        Args:
            n_components (int): The new number of components to use.
        """
        self.n_components = n_components
        self._fit()

    def set_data(self, data):
        """
        Sets new data for the PCA model and refits the model.

        Args:
            data (np.ndarray): The new data to use.
        """
        self.data = data
        self._fit()

    def optimal_num_components(self, threshold=0.95):
        """
        Determines the optimal number of principal components to reach the desired variance threshold.

        Args:
            threshold (float): Cumulative variance threshold to reach. Defaults to 0.95.

        Returns:
            int: Optimal number of principal components.
        """
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        num_components = np.argmax(cumulative_variance >= threshold) + 1
        return num_components

    def get_transformed_data_2d(self):
        """
        Get the transformed data in 2D space.

        Returns:
            array: Transformed data in 2D.
        """
        if self.n_components < 2:
            raise ValueError("Need at least 2 components for 2D data.")

        return self.transformed_data[:, :2]

    def get_transformed_data_3d(self):
        """
        Get the transformed data in 3D space.

        Returns:
            array: Transformed data in 3D.
        """
        if self.n_components < 3:
            raise ValueError("Need at least 3 components for 3D data.")

        return self.transformed_data[:, :3]

    def _calculate_cosine_similarity(self, transformed_data):
        """
        Calculate the cosine similarity for each point in transformed data.
        Cosine similarity is used to measure quality of representation.
        """
        norms = np.linalg.norm(transformed_data, axis=1)
        cosine_similarity = np.abs(transformed_data) / norms[:, None]
        return cosine_similarity

    def plot_2d(self, point_size=50):
        """
        Plot the transformed data in 2D space with a colormap indicating the quality of representation.
        
        Args:
            labels (array-like, optional): Labels for each data point.
            point_size (float, optional): Size of the points in the scatter plot.
        """
        transformed_data = self.get_transformed_data_2d()
        cosine_similarity = self._calculate_cosine_similarity(transformed_data)
        quality_representation = cosine_similarity[:, 0]  # Quality with respect to the first PC

        plt.figure()
        scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1],
                              c=quality_representation, cmap='brg', s=point_size,
                              norm=Normalize(vmin=0, vmax=1))
        plt.colorbar(scatter, label='Cosine Similarity with PC1')

        # Adding explained variance to axis labels
        evr = self.explained_variance_ratio
        plt.xlabel(f'Principal Component 1 ({evr[0]*100:.2f}%)')
        plt.ylabel(f'Principal Component 2 ({evr[1]*100:.2f}%)')

        plt.title('2D PCA with Quality Representation Colormap')


    def plot_3d(self, point_size=50):
        """
        Plot the transformed data in 3D space with a colormap indicating the quality of representation.
        
        Args:
            labels (array-like, optional): Labels for each data point.
            point_size (float, optional): Size of the points in the scatter plot.
        """
        transformed_data = self.get_transformed_data_3d()
        cosine_similarity = self._calculate_cosine_similarity(transformed_data)
        quality_representation = cosine_similarity[:, 0]  # Quality with respect to the first PC

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2],
                             c=quality_representation, cmap='brg', s=point_size,
                             norm=Normalize(vmin=0, vmax=1))
        fig.colorbar(scatter, ax=ax, label='Cosine Similarity with PC1')

        # Adding explained variance to axis labels
        evr = self.explained_variance_ratio
        ax.set_xlabel(f'PC 1 ({evr[0]*100:.2f}%)')
        ax.set_ylabel(f'PC 2 ({evr[1]*100:.2f}%)')
        ax.set_zlabel(f'PC 3 ({evr[2]*100:.2f}%)')

        ax.set_title('3D PCA with Quality Representation Colormap')

    def plot_correlation_circle(self):
        """
        Plots the correlation circle showing the projection of the original variables on the principal components.
        """
        fig, ax = plt.subplots()
        for i in range(self.correlations.shape[0]):
            ax.arrow(0, 0, self.correlations[i, 0], self.correlations[i, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')
            ax.text(self.correlations[i, 0], self.correlations[i, 1], f'Var{i+1}', color='red')

        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, color='blue', fill=False)
        ax.add_artist(circle)

        # Set the same scale for both axes and limit them to [-1, 1]
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal', 'box')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Correlation Circle')

    def plot_distance_from_origin(self):
        """
        Plots a bar graph of the distance of each point from the origin in PCA space.
        """
        distances = np.sqrt(np.sum(self.transformed_data**2, axis=1))

        plt.figure()
        plt.bar(range(len(distances)), distances)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance from Origin')
        plt.title('Distance from Origin in PCA Space')

    def plot_explained_variance(self):
        """
        Plots a bar graph of the explained variance by each principal component.
        """
        plt.figure()
        plt.bar(range(len(self.explained_variance_ratio)), self.explained_variance_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by PCA Components')

    def plot_scree(self):
        """
        Plots the scree plot of the PCA analysis.

        The scree plot displays the cumulative explained variance by each component of the PCA. 
        It helps in determining the number of principal components to retain by showing how much 
        variance is captured by each successive principal component.
        """
        plt.figure()
        plt.plot(range(len(self.explained_variance_ratio)+1), [0] + list(np.cumsum(self.explained_variance_ratio)))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
