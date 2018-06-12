import cv2
import os
import shutil 

def write_to_clusters(labels):
    print("Writing clusters to cluster folder")
    i = 0

    with open('text_files/list_of_files.txt') as f:
        images = f.read().splitlines()

    for image in images:
        dst = "./clusters/cluster_%d" % labels[i]
        if not os.path.exists(dst):
            os.makedirs(dst)
        shutil.copyfile("./"+image, dst+"/"+os.path.basename(image))
        i += 1  