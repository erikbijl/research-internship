from sklearn.cluster import MeanShift, estimate_bandwidth
from distanceTable import getDistanceTable

[data, distances] = getDistanceTable()

bandwidth = estimate_bandwidth(distances, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(distances)
labels = ms.labels_
	
for i in range(len(data["faces"])):
	print(data["faces"][i]["file"])
	print(labels[i])

