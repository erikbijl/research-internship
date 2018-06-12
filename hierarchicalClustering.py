from write_to_clusters import write_to_clusters
from evaluate import f_measure
import numpy as np
import shutil
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#merge a list of 
def merge(l):
	out = []
	while len(l)>0:
	    first, *rest = l
	    first = set(first)

	    lf = -1
	    while len(first)>lf:
	        lf = len(first)

	        rest2 = []
	        for r in rest:
	            if len(first.intersection(set(r)))>0:
	                first |= set(r)
	            else:
	                rest2.append(r)     
	        rest = rest2

	    out.append(first)
	    l = rest
	return out

thres = list(map(lambda x: x/100, range(70,90,2)))
f_score = []

dist_tbl = np.loadtxt("text_files/distance_table.txt")
amt = len(dist_tbl)

for threshold in thres:
	print(str(threshold))
	merges = []
	for i in range(0,amt): 
		merges.append([i])
		for j in range(i,amt): 
			if dist_tbl[i][j] < threshold:
				merges.append([i,j])


	sets = merge(merges)	
	labels = np.zeros(amt)

	print("calculate labels")
	for i in range(0, amt):
		idx = 0
		for subset in sets:
			if i in subset:	
			   	labels[i] = idx
			   	break
			idx += 1

	write_to_clusters(labels)
	#f_score.append(f_measure())
	f_score.append(f_measure())

	#clean directory
	for f in os.listdir("./clusters/"):
		shutil.rmtree(os.path.join("./clusters/", f))


plt.plot(thres, f_score, 'r-')
plt.xlabel('Value of threshold')
plt.ylabel('F-measure')
plt.title('The F-measure over the threshold in threshold clustering')
plt.savefig('f_over_threshold_threshold_clustering.png')