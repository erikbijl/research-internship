import sys
import numpy as np
from write_to_clusters import write_to_clusters
from evaluate import f_measure
from sklearn.cluster import MeanShift, estimate_bandwidth

emb = np.loadtxt("text_files/embeddings.txt")
x = float(sys.argv[1])
#perform the meanshift clustering 
ms = MeanShift(bandwidth=x)
ms.fit(emb)
labels = ms.labels_
lbl_amt = len(set(labels))

#write clusters
write_to_clusters(labels)

#calculate f_score
	
f_measure()