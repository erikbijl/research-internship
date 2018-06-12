from write_to_clusters import write_to_clusters
from evaluate import f_measure
import numpy as np 
import os
import shutil
import pyflann
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def build_index(dataset, n_neighbors):
    """
    Takes a dataset, returns the "n" nearest neighbors
    """
# Initialize FLANN
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset,algorithm='kdtree',trees=4)
    #print params
    nearest_neighbors, dists = flann.nn_index(dataset, n_neighbors, checks=params['checks'])
    return nearest_neighbors, dists

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

#calculate a distance measure between the faces
def distance_function(nn_a, nn_b, k):
	d = 0
	rank_b_in_a = k
	if nn_b[0]in nn_a:
		rank_b_in_a = int(np.argwhere(nn_a == nn_b[0]))
	set_x = set(nn_a[:rank_b_in_a]).difference(set(nn_b[:rank_b_in_a]))
	return len(set_x)

dist_tbl = np.loadtxt("text_files/distance_table.txt")
embeddings = np.loadtxt("text_files/embeddings.txt")
print(dist_tbl)	
cluster_folder = "./clusters/"

thres = list(map(lambda x: x/10, range(5,6)))
#thres = range(0,1)	0
#k = 3	
#K = range(, 10, 1)
K = range(51,101)

#f_score = []
f_score_k = []

#amount of faces in distance matrix 
amt = len(dist_tbl)

print("create nearest neighbors")
app_nearest_neighbors, dists = build_index(embeddings, n_neighbors=len(embeddings))

print("create rankings")
rankings = np.zeros((amt, amt))
for i in range(0, amt):
	for j in range(0,amt):
		rankings[i][j] = np.argwhere(app_nearest_neighbors[i] == j)
print(rankings) 

for k in K:
	print("K=="+str(k))
	print("create symmetric distance matrix")
	#create distance matrix 
	matrix = np.zeros((amt, amt))
	for i in range(0,amt):
		for j in range(0,amt):
			matrix[i][j] = distance_function(app_nearest_neighbors[i][:k+1], app_nearest_neighbors[j][:k+1], k)	

	print("create assymmetric distance matrix")
	distance = np.zeros((amt, amt))
	
	for i in range(0,amt):
		for j in range(0,amt):
			if (not i == j):
				distance[i][j] = (matrix[i][j] + matrix[j][i]) / min(k, rankings[i][j], rankings[j][i])
	
	print("calculate merges") 	
	for threshold in thres:
		merges = []
		print(threshold)
		#amount of first elements
		for i in range(0,amt):
			merges.append([i])	
			for j in range(0,amt):
				if(distance[i][j] <= threshold):
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
		f_score_k.append(f_measure())

		#clean directory
		for f in os.listdir(cluster_folder):
			shutil.rmtree(os.path.join(cluster_folder, f))

plt.plot(K, f_score_k, 'r-')
plt.xlabel('Value of k')
plt.ylabel('F-measure')
plt.title('The F-measure as a result over in ARO-clustering')
plt.savefig('f_over_k_ARO.png')

# plt.plot(thres, f_score, 'r-')
# plt.xlabel('Value of threshold')
# plt.ylabel('F-measure')
# plt.title('The F-measure as a result of the threshold used in ARO-clustering')
# plt.savefig('f_over_threshold_ARO.png')