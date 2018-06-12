import sys
import numpy as np
from write_to_clusters import write_to_clusters
from evaluate import f_measure
from sklearn.cluster import DBSCAN

emb = np.loadtxt("text_files/embeddings.txt")
x = float(sys.argv[1])

#perform the meanshift clustering 
db = DBSCAN(eps=x, min_samples=2).fit(emb)
labels = db.labels_
#write clusters
write_to_clusters(labels)

#calculate f_score
	
f_measure()