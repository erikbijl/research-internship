import numpy as np
from write_to_clusters import write_to_clusters
from sklearn.cluster import MeanShift
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

print("Amount of folders in images:")
print(len(os.listdir('./images/')))

X = []
amount_of_labels = []
scores = []

for x in range(1, 100):
	x = x /100
	print(x)
	X.append(x)

	#perform the meanshift clustering 
	ms = MeanShift(bandwidth=x)
	ms.fit(emb)
	labels = ms.labels_
	lbl_amt = len(set(labels))
	amount_of_labels.append(lbl_amt)
	
	#write clusters
	write_to_clusters(labels)
	
	#calculate f_score
	scores.append(f_measure())
	
	#clean directory
	for f in os.listdir(cluster_folder):
		shutil.rmtree(os.path.join(cluster_folder, f))
plt.figure(1)
plt.plot(X, amount_of_labels, 'b-')
plt.xlabel('Bandwith')
plt.ylabel('Amount of clusters')
plt.title('The amount of clusters as a result of the bandwidth')
plt.savefig('clusters_over_bandwidth.png')

plt.figure(2	)
plt.plot(X, scores, 'r-')
plt.xlabel('Bandwith')
plt.ylabel('F-measure')
plt.title('The F-measure as a result of the bandwidth')
plt.savefig('f_over_bandwidth.png')