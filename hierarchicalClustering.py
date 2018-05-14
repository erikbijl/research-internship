from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from distanceTable import getDistanceTable

[data, distances] = getDistanceTable()

Z = linkage(distances, 'ward')
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
	Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()