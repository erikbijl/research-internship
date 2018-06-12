import numpy as np
from write_to_clusters import write_to_clusters
from sklearn.cluster import KMeans
import os
import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from evaluate import f_measure

cluster_folder = "./clusters/"
emb = np.loadtxt("text_files/embeddings.txt")

k = 100

print(k)
#perform a kmeans clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(emb)
labels = kmeans.labels_

#write clusters
write_to_clusters(labels)

#show f_score
f_measure()

#clean directory
for f in os.listdir(cluster_folder):
	shutil.rmtree(os.path.join(cluster_folder, f))