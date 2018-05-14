# Research-internship: face clustering

In this repository multiple face clustering methods are implemented. These methods work on top of the facenet implementation provided by https://github.com/pavelkrolevets/face_clustering. 

### storeFaces.py
This file extracts faces from images stored in the folder /images. The faces containing the file name, id and embedding are stored in a Json file called data.json. 

### distanceTable.py 
Has a function getDistanceTable that reads the faces in the Json file and returns a distance matrix containing all distances between faces in the file. 

### thresholdClustering.py
Clusters the faces from the image folder to a cluster folder. If a difference between faces is under a certain threshold the faces are stored in the same cluster.

### kmeans.py
Simple implementation of Kmeans algorithm on the distance matrix.

### meanshift.py
Simple implementation of the mean shift clustering algorithm on the distance matrix.

### hierarchicalClustering.py
Simple implementation of a dendrogram calculation on the distance matrix. 
