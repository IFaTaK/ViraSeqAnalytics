import sys
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from src.align_genome import read_sequence
from src.genome import UK,CH
from src.extraction import data_from_genome
from src.pca import PCAAnalysis
from src.cluster_detection import KMeansClustering
from tqdm import tqdm


def process_read_sequence(args):
    name, sequence = args
    return name, read_sequence(sequence, 1000, 12)

def main():
    NUMBER_GENOME = len(UK) + len(CH)
    UK_aligned, CH_aligned = {}, {}
    print("")
    with tqdm(total=NUMBER_GENOME, desc="Reading and aligning genomes:") as pbar:
        with Pool(processes=4) as pool:  # Limit consumption to a maximum of 4 CPU cores
            UK_results = []
            for result in pool.imap_unordered(process_read_sequence, UK.items()):
                UK_results.append(result)
                pbar.update()

        with Pool(processes=4) as pool:  # Limit consumption to a maximum of 4 CPU cores
            CH_results = []
            for result in pool.imap_unordered(process_read_sequence, CH.items()):
                CH_results.append(result)
                pbar.update()

    UK_aligned = {name: result for name, result in UK_results}
    CH_aligned = {name: result for name, result in CH_results}

    data = []
    print("\n")
    with tqdm(total=NUMBER_GENOME, desc="Comparing aligned genome with reference sequence (matchrate):") as pbar:
        for sequence in UK_aligned.values():
            data.append(data_from_genome(sequence))
            pbar.update()

        for sequence in CH_aligned.values():
            data.append(data_from_genome(sequence))
            pbar.update()

    data = np.array(data)

    print("\nPCA on the dataset:")
    pca = PCAAnalysis(data, n_components=3)

    data_2d, data_3d = pca.get_transformed_data_2d(), pca.get_transformed_data_3d()

    fig = plt.figure(figsize=(18, 6))
    kmeans = KMeansClustering(data,2)
    kmeans.fit()
    labels = kmeans.labels

    print("K-Means Clustering on the dataset:")
    ax1 = fig.add_subplot(121) # 1 row, 2 columns, 1st subplot
    ax1.scatter(data_2d[:,0],data_2d[:,1], c=labels, s=80, cmap='coolwarm')
    for k in range(np.shape(data_2d[:,0])[0]):
        ax1.text(data_2d[k, 0], data_2d[k, 1], str(k+1),fontsize = 13, weight = "bold")
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.spines['top'].set_position(('data',0))
    ax1.spines['right'].set_position(('data',0))
    ax1.grid(True)
    ax1.set_title('2D K-Means Clustering with k=2')

    ax2 = fig.add_subplot(122, projection='3d') # 1 row, 2 columns, 2nd subplot
    ax2.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, s=80, cmap='coolwarm', alpha=1)
    for k in range(np.shape(data_3d[:,0])[0]):
        ax2.text(data_3d[k, 0], data_3d[k, 1], data_3d[k, 2],str(k+1),fontsize = 13, weight = "bold")
    ax2.set_title('3D K-Means Clustering with k=2')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    plt.show()

if __name__ == "__main__":
    main()
