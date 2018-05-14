from sklearn.cluster import KMeans
from distanceTable import getDistanceTable

[data, distances] = getDistanceTable()

kmeans = KMeans(n_clusters=4, random_state=40, max_iter=100000)
kmeans.fit(distances)
labels_kmeans = kmeans.labels_

for i in range(len(data["faces"])):
	print(data["faces"][i]["file"])
	print(labels_kmeans[i])