import numpy as np
from write_to_clusters import write_to_clusters
from sklearn.cluster import DBSCAN
import os
import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from evaluate import f_measure

cluster_folder = "./clusters/"
emb = np.loadtxt("text_files/embeddings.txt")

# The following bandwidth can be automatically detected using
#bandwidth = estimate_bandwidth(emb, quantile=0.25, n_samples=1000)
#print(bandwidth)

K = list(map(lambda x: x/100, range(1,150)))
	
scores = []

for k in K:
	print(k)
	#perform a kmeans clustering
	db = DBSCAN(eps=k, min_samples=2).fit(emb)
	labels = db.labels_
	#write clusters
	write_to_clusters(labels)
	
	#calculate f_score
	scores.append(f_measure())
	
	#clean directory
	for f in os.listdir(cluster_folder):
		shutil.rmtree(os.path.join(cluster_folder, f))

plt.plot(K, scores, 'r-')
plt.xlabel('Maximum distance')
plt.ylabel('F-measure')
plt.title('The F-measure as a result of the maximum distance between examples')
plt.savefig('DBSCAN_f_measure.png')