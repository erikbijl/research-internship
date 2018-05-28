# MIT License
#
# Copyright (c) 2017 PXL University College
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import detect_face
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import facenet
from sklearn.cluster import KMeans
import cv2

def main(args):
    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model(args.model)
            print("Loading faces")
            [image_names, image_list] = load_images_from_folder(args.data_dir)


            print("Aligning faces")
            images = align_data(image_list, args.image_size, args.margin, pnet, rnet, onet)

            print("Calculating embeddings")
            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(image_list)
            print("Amount of images: "+str(nrof_images))
            matrix = np.zeros((nrof_images, nrof_images))
            
            print('')
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist
                    print('  %1.4f  ' % dist, end='')
                print('')

            print('')  

            #perform a kmeans clustering
            kmeans = KMeans(n_clusters=2, random_state=40, max_iter=100000)
            kmeans.fit(matrix)
            labels_kmeans = kmeans.labels_

            i = 0
            for image in image_names:
                print(image)
                img = cv2.imread(image)
                target_directory = "/home/erik/Documenten/Msc/Research internship/research-internship/clusters/cluster_%d" % labels_kmeans[i]
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)
                cv2.imwrite(target_directory+"/"+os.path.basename(image), img)
                cv2.waitKey(0)
                i += 1              


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []

    for x in range(len(image_list)):    
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images_name = []
    images_data = []
    for subfolder in os.listdir(folder):
        for filename in os.listdir(os.path.join(folder, subfolder)):
            file_path = os.path.join(os.path.join(folder, subfolder), filename)
            images_name.append(file_path)
            img = misc.imread(file_path)
            if img is not None:
                images_data.append(img)
    return [images_name, images_data] 


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('data_dir', type=str,
                        help='The directory containing the images to cluster into folders.')
    parser.add_argument('out_dir', type=str,
                        help='The output directory where the image clusters will be saved.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=1)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--largest_cluster_only', action='store_true',
                        help='This argument will make that only the biggest cluster is saved.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))