import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import random
import time
from object_detection.utils import label_map_util

# Load Graph
def load_and_create_graph(path_to_pb) :
    """
    Loads pre-trained graph from .pb file.
    path_to_pb: path to saved .pb file
    Tensorflow keeps graph global so nothing is returned
    """
    with tf.compat.v1.gfile.FastGFile(path_to_pb,'rb') as f:
        # Initialize graph drefinition
        graph_def = tf.compat.v1.GraphDef()
        # reads file
        graph_def.ParseFromString(f.read())
        # imports as tf.graph
        _ = tf.compat.v1.import_graph_def(graph_def, name='')


def read_cv_image(filename) :
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_mpl_img_with_detections(img, dets, scores,
                                 classes, category_index,
                                 thres=0.6) :
    """
    Applies thresholding to each box score and
    plot bbox results on image.
    img: input image as numpy array
    dets: list of K detection outputs for given image.(size:[1,K])
    scores: list of detection score for each detection output(size: [1,K]).
    classes: list of predicted class index(size: [1,K])
    category_index: dictionary containing mapping from class index to class name.
    thres: threshold to filter detection boxes:(default: 0.6)
    By default K:100 detections
    """
    # plotting utilities from matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    # To use common color of one class and different for different classes
    colors = dict()
    # iterate over all proposed bbox
    # choo