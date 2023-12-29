import sys
import matplotlib.pyplot as plt
import numpy as np
from src.align_genome import read_sequence
from src.genome import UK,CH
from src.extraction import data_from_genome
from src.pca import PCAAnalysis
from src.cluster_detection import KMeansClustering

UK_aligned, CH_aligned = {}, {}
NUMBER_GENOME = len(UK)+len(CH)
i = 0
print("Genome read and alignment:")
for name, sequence in UK.items():
    loading_bar = '#' * int(30 * (i / (NUMBER_GENOME-1))) + '-' * (30 - int(30 * (i / (NUMBER_GENOME-1))))
    sys.stdout.write(f"\r[{loading_bar}]")
    sys.stdout.flush()
    UK_aligned[name] = read_sequence(sequence, 1000, 12)
    i += 1

for name,sequence in CH.items():
    loading_bar = '#' * int(30 * (i / (NUMBER_GENOME-1))) + '-' * (30 - int(30 * (i / (NUMBER_GENOME-1))))
    sys.stdout.write(f"\r[{loading_bar}]")
    sys.stdout.flush()
    CH_aligned[name] = read_sequence(sequence, 1000, 12)
    i += 1
sys.stdout.write(f"\r[{loading_bar}]")
sys.stdout.flush()

data = []
i = 0
print("\nComparing aligned genome with reference sequence (matchrate):")
for sequence in UK_aligned.values():
    loading_bar = '#' * int(30 * (i / (NUMBER_GENOME-1))) + '-' * (30 - int(30 * (i / (NUMBER_GENOME-1))))
    sys.stdout.write(f"\r[{loading_bar}]")
    sys.stdout.flush()
    data.append(data_from_genome(sequence))
    i += 1

for sequence in CH_aligned.values():
    loading_bar = '#' * int(30 * (i / (NUMBER_GENOME-1))) + '-' * (30 - int(30 * (i / (NUMBER_GENOME-1))))
    sys.stdout.write(f"\r[{loading_bar}]")
    sys.stdout.flush()
    data.append(data_from_genome(sequence))
    i += 1
sys.stdout.write(f"\r[{loading_bar}]")
sys.stdout.flush()

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
