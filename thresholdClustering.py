from facematch_functions import getFace, getEmbedding
import numpy as np
import cv2
from PIL import Image
import os

images_folder = "/home/erik/Documenten/Msc/Research internship/facematch/images"
clusters_folder = "/home/erik/Documenten/Msc/Research internship/facematch/clusters"

def compareFaces(img_path, cluster_amt):
    print("Clustering:"+img_path)
    for fname in os.listdir(clusters_folder):
        path = os.path.join(clusters_folder, fname)
        if os.path.isdir(path):
            for img in os.listdir(path):
                path_to_file = path+"/"+img
                img1 = cv2.imread(path_to_file)
                img2 = cv2.imread(img_path)
                face1 = getFace(img1)
                face2 = getFace(img2)
                if(face1 and face2):
                    dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
                    if(dist<1.1):
                        print("Match found: "+path)
                        cv2.imwrite(path+"/"+os.path.basename(img_path), img2)
                        cv2.waitKey(0)
                        return cluster_amt

    cluster_amt += 1
    new_directory = clusters_folder+"/"+"cluster_%d" % cluster_amt
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        img2 = cv2.imread(img_path) 
        cv2.imwrite(new_directory+"/"+os.path.basename(img_path), img2)
        cv2.waitKey(0)
        print("No match, created new cluster: "+new_directory)
    return cluster_amt

cluster_amt = 0
for path in os.listdir(images_folder):
    directory = os.path.join(images_folder, path)
    if os.path.isdir(directory):
            for img in os.listdir(directory):
                cluster_amt = compareFaces(os.path.join(directory,img), cluster_amt)