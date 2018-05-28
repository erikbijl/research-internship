import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from distanceTable import getDistanceTable

[data, distances] = getDistanceTable()

bandwidth = estimate_bandwidth(distances)

ms = MeanShift()
ms.fit(distances)
labels = ms.labels_
	
for i in range(len(data["faces"])):
	print(os.path.basename(data["faces"][i]["file"]))
	print(labels[i])