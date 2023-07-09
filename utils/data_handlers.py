import sys

package_paths = [
    './yolocode/',
]
for pth in package_paths:
    sys.path.append(pth)

import re
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
# add path
import colorsys
from PIL import Image, ImageDraw, ImageFont
import random
from ..yolocode.yolo3.model import  preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_correct_boxes
import PIL
import torch
# !pip install torchmetrics
import shutil
from pprint import pprint
import torchmetrics.detection.mean_ap as MeanAveragePrecision


from ..yolocode.yolo3.utils import *


# 1
def make_annotation_txtfile():
    '''This func read image_annotations.csv ['id','file_name','height','width','annotations']
        and split data into unlable pool and test data
    '''
    # select nrows from csv file - number of images in dataset for active learning
    image_annotations = pd.read_csv("./coco_dataset/image_annotations.csv", nrows=10000)
    print('image_annotations', image_annotations.size)

    file = open("coco_annotation.txt", "a")
    for index, row in image_annotations.iterrows():
        a = row["annotations"]
        a = re.sub(r" ", "", a)
        a = re.sub(r"],", " ", a)
        a = re.sub(r"\[", "", a)
        a = re.sub(r"\]", "", a)
        if a != "":
            # file_path = os.path.join(path, row["file_name"])
            file.writelines(row["file_name"] + " " + a + "\n")

    # split into train and test
    file = open("coco_annotation.txt", "r")
    a = len(file.readlines())
    train_num = int(a * 0.8)  ## Split data into 80 and 20 percent of select data
    test_num = a - train_num
    file.close()
    train_file = open("coco_unlabel_pool.txt", "a")
    test_file = open("coco_test.txt", "a")
    with open("coco_annotation.txt", "r") as file:
        for i, line in enumerate(file):
            if i <= train_num:
                train_file.writelines(line)
            else:
                test_file.writelines(line)
    train_file.close()
    test_file.close()
    file.close()


# 2
# select top n training data
def select_train_data(n=100, append=False):
    ''' This function select training images from unlabel pool.  '''
    temp_file = open("temp.txt", "a+")
    if append:
        train_file = open("coco_train.txt", "a+")
    else:
        train_file = open("coco_train.txt", "w")
    pool_file = open("coco_unlabel_pool.txt", "r")
    a = pool_file.readlines()

    for i in range(n):
        train_file.writelines(a[i])
    for i in range(n, len(a)):
        temp_file.writelines(a[i])

    pool_file.close()
    temp_file.close()
    train_file.close()
    if os.path.exists("coco_unlabel_pool.txt"):
        os.remove("coco_unlabel_pool.txt")
        os.rename('temp.txt', 'coco_unlabel_pool.txt')


# 3
def move_to_train(selected):
    temp_file = open("temp.txt", "a+")
    train_file = open("coco_train.txt", "a+")
    pool_file = open("coco_unlabel_pool.txt", "r")
    lines = pool_file.readlines()

    for line in lines:
        name = line.split()[0]
        if name in selected:
            train_file.writelines(line)
        else:
            temp_file.writelines(line)

    pool_file.close()
    temp_file.close()
    train_file.close()
    if os.path.exists("coco_unlabel_pool.txt"):
        os.remove("coco_unlabel_pool.txt")
        os.rename('temp.txt', 'coco_unlabel_pool.txt')


# 4
def get_unlabel_images():
    """ This function return names of images in unlable pool. """
    pool_file = open("coco_unlabel_pool.txt", "r")
    lines = pool_file.readlines()
    unlable_images = []
    for i in lines:
        unlable_images.append(i.split()[0])

    return unlable_images


# 5
def annotated_images(model_path="./yolocode/model_data/yolo_weights.h5", n=1000, score_type='normal'):
    """ This function make prediction on all the images in unlabel pool and
    select training images form unlabel pool based of images scores """
    # load trained model
    model = load_trained_model(input_shape=(416, 416), num_classes=80, load_pretrained=True, freeze_body=2,
                               weights_path=model_path)

    unlable_images = get_unlabel_images()

    # calculate score for images
    images_score = np.empty((0))
    iter = 0
    for i in chunks(unlable_images, 10):
        if iter * 10 >= n:
            break
        if iter % 10 == 0:
            print('progress iteration ', iter)
        iter = iter + 1
        batch = np.empty((0, 416, 416, 3))
        for img in i:
            image = read_image(img)
            image = preprocess_image(image)
            batch = np.append(batch, image, axis=0)
        score, prob_score = get_score(batch, model)
        #         print('prob_score', prob_score)
        if score_type == 'prob':
            images_score = np.append(images_score, prob_score)
        elif score_type == 'normal':
            images_score = np.append(images_score, score)

    # select top n images
    top_score = (images_score).argsort()[:n]
    selected = []
    for i in top_score:
        selected.append(unlable_images[i])
    move_to_train(selected)


