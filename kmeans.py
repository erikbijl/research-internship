import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn.cluster import KMeans
from distanceTable import getDistanceTable

clusters_folder = "/home/erik/Documenten/Msc/Research internship/research-internship/clusters"

def findAmountOfClusters(amountOfFaces, X):
    # k means determine k
    distortions = [0]
    diff = []
    K = range(1,amountOfFaces)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(spatial.distance.cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        if(distortions[k-1] > 1.08*distortions[k]):
        	elbow = float (distortions[k-1]) / float (distortions[k]) 
        	diff.append(elbow)
    # Plot the elbow
	# plt.plot(K, distortions[1:], 'bx-')
	#plt.xlabel('k')
    #plt.ylabel('Distortion')
    #plt.title('The Elbow Method showing the optimal k')
    #plt.show()

    return diff.index(max(diff))

[data, distances] = getDistanceTable()

nr_faces = findAmountOfClusters(len(data["faces"]), distances)

print("Expected number of clusters is: %d"%nr_faces)

kmeans = KMeans(n_clusters=nr_faces, random_state=40, max_iter=100000)
kmeans.fit(distances)
labels_kmeans = kmeans.labels_

for i in range(len(data["faces"])):
    print(os.path.basename(data["faces"][i]["file"]))
    print("Cluster: " + str(labels_kmeans[i]))
    img = cv2.imread(data["faces"][i]["file"])
    target_directory = clusters_folder+"/"+"cluster_%d" % labels_kmeans[i]
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    cv2.imwrite(target_directory+"/"+os.path.basename(data["faces"][i]["file"]), img)
    cv2.waitKey(0)        