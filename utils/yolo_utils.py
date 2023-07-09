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
import numpy as np

from training_functions import *

# preprocess
# 1
def read_image(image_name):

    path = "./coco_dataset/images/"
    image = PIL.Image.open(path + image_name)
    return image

# 2
def preprocess_image(image):
    sq_image = letterbox_image(image, (416 ,416))
    image_data = np.array(sq_image)
    image_data = image_data/255.0
    image_data = np.expand_dims(image_data, 0) # (1, 416, 416, 3)
    return image_data

# 3
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# 4
def load_trained_model(input_shape=(416, 416), num_classes=80, load_pretrained=True, freeze_body=2,
                       weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    # anchors_path = '../input/yolocode/model_data/yolo_anchors.txt'

    anchors = get_anchors()

    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    ## create list of o/p shape -
    ## shape=(None, 13, 13, num_anchors/3 , num_classes+5);
    ## shape=(None, 26, 26, num_anchors/3, num_classes+5);
    ## shape=(None, 52, 52, num_anchors/3, num_classes+5)
    ## +5 for [xmin,ymin,xmax,ymax,class] for all classes and all anchore div into 3

    # load model architecture
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # LOAD pretarin model weights

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    return model_body


# 5
def get_score(image, yolo_model):
    """ This function calculate score of the image """
    # get data
    anchors = get_anchors()
    class_names = get_classes()
    num_anchors = len(anchors)
    num_classes = len(class_names)

    yolo_outputs = yolo_model.predict(image)
    # (1, 13, 13, 255)
    # (1, 26, 26, 255)
    # (1, 52, 52, 255)

    score_batch = np.zeros((image.shape[0]))  # ((None,416,416,3))
    al_ep_score = np.zeros((image.shape[0]))
    for i in range(len(image)):
        _output = [np.expand_dims(yolo_outputs[0][i], 0), np.expand_dims(yolo_outputs[1][i], 0),
                   np.expand_dims(yolo_outputs[2][i], 0)]
        out_boxes, out_scores, out_classes, uncert_all_classes = yolo_eval(_output, anchors, num_classes,
                                                                           [416, 416],
                                                                           max_boxes=20, score_threshold=.6,
                                                                           iou_threshold=.5)

        # calculating aleatoric and epistemic uncertainities for every image
        #         print('out_classes', out_classes)
        #         print('out_scores', out_scores)
        #         print('uncert', uncert_all_classes)
        u_al = 0
        u_ep = 0
        normalized_scores = out_scores
        if len(out_scores) > 0:
            normalized_scores = out_scores / np.max(out_scores)
            normalized_scores = normalized_scores.numpy()
        out_classes = out_classes.numpy()
        #         print('normalized', normalized_scores)
        sum_pi_mu = 0
        for c, s in zip(out_classes, normalized_scores):
            for b in uncert_all_classes[c]:
                for u in b:
                    #                     print('s', s, 'mu', u[0])
                    sum_pi_mu += np.log(
                        s * tf.clip_by_value(u[0], clip_value_min=0.0, clip_value_max=tf.float32.max) + 1)

        for c, s in zip(out_classes, normalized_scores):
            for b in uncert_all_classes[c]:
                for u in b:
                    mu = u[0]
                    var = u[1]
                    #                     print('mu', mu, 'var', var)
                    u_al += np.log(s * var + 1)
                    u_ep += np.log(s * np.abs(mu - sum_pi_mu) ** 2 + 1)

        score_batch[i] = np.average(out_scores)
        #         print('u_al', u_al, 'u_ep', u_ep)
        al_ep_score[i] = u_al + u_ep
    return score_batch, al_ep_score


# 6
def yolo_predict(image, model_path="./yolocode/model_data/yolo_weights.h5"):
    # preprocess image
    image_data = preprocess_image(image)  # (1, 416, 416, 3)

    # get data
    anchors = get_anchors()
    class_names = get_classes()
    num_anchors = len(anchors)
    num_classes = len(class_names)

    # load model
    yolo_model = load_trained_model(input_shape=(416, 416), num_classes=num_classes, load_pretrained=True,
                                    freeze_body=2, weights_path=model_path)
    # yolo_model = load_model(model_path, compile=False)

    yolo_outputs = yolo_model.predict(image_data)
    # (1, 13, 13, 255)
    # (1, 26, 26, 255)
    # (1, 52, 52, 255)

    out_boxes, out_scores, out_classes, _ = yolo_eval(yolo_outputs, anchors, num_classes,
                                                      [image.size[1], image.size[0]], max_boxes=20, score_threshold=.6,
                                                      iou_threshold=.5)
    image = draw_boxes(out_boxes, out_scores, out_classes, image)
    return image


# 7
def get_uncert_score(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), name=None):
    """ This function calculate uncertanty score before applying nms """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores
    order = tf.argsort(scores)
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    nms_idx = []
    uncert = []
    while len(order) > 0:
        idx = order[-1]

        keep.append(boxes[idx].numpy())
        nms_idx.append(idx.numpy())
        order = order[:-1]

        xx1 = tf.gather(x1, indices=order, axis=0)
        xx2 = tf.gather(x2, indices=order, axis=0)
        yy1 = tf.gather(y1, indices=order, axis=0)
        yy2 = tf.gather(y2, indices=order, axis=0)

        xx1 = tf.math.maximum(xx1, x1[idx])
        yy1 = tf.math.maximum(yy1, y1[idx])
        xx2 = tf.math.minimum(xx2, x2[idx])
        yy2 = tf.math.minimum(yy2, y2[idx])

        w = xx2 - xx1
        h = yy2 - yy1

        w = tf.clip_by_value(w, clip_value_min=0.0, clip_value_max=tf.float32.max)
        h = tf.clip_by_value(h, clip_value_min=0.0, clip_value_max=tf.float32.max)

        inter = w * h

        rem_areas = tf.gather(areas, indices=order, axis=0)

        union = (rem_areas - inter) + areas[idx]

        IoU = inter / union

        #         mask = IoU < iou_threshold
        #         order = order[mask]
        mask = IoU < 0.5
        inv_mask = IoU > 0.5

        cluster_idx = order[inv_mask]
        cluster_idx = list(cluster_idx.numpy())
        cluster_idx.append(idx.numpy())
        #         tf.stack([cluster_idx.numpy(), [idx.numpy()]], axis=0)
        order = order[mask]

        x1_ = [x1[i].numpy() for i in cluster_idx]
        y1_ = [y1[i].numpy() for i in cluster_idx]
        x2_ = [x2[i].numpy() for i in cluster_idx]
        y2_ = [y2[i].numpy() for i in cluster_idx]

        x1_ = np.array(x1_)
        y1_ = np.array(y1_)
        x2_ = np.array(x2_)
        y2_ = np.array(y2_)

        x1_mu = x1_.mean()
        y1_mu = y1_.mean()
        x2_mu = x2_.mean()
        y2_mu = y2_.mean()

        x1_var = x1_.var()
        y1_var = y1_.var()
        x2_var = x2_.var()
        y2_var = y2_.var()

        uncert.append([[x1_mu, x1_var], [y1_mu, y1_var], [x2_mu, x2_var], [y2_mu, y2_var]])

    return tf.convert_to_tensor(keep, dtype=tf.float32), tf.convert_to_tensor(nms_idx, dtype=tf.int32), uncert


# 8
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    uncert_all_classes = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        mynms_out, mynms_index, uncert = get_uncert_score(class_boxes, class_box_scores, max_boxes_tensor,
                                                          iou_threshold=iou_threshold)
        uncert_all_classes.append(uncert)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_, uncert_all_classes


# 9
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# 10
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, np.float32)

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


# 11
def draw_boxes(out_boxes, out_scores, out_classes, image):
    class_names = get_classes()
    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300
    colors = get_colors_for_classes(len(class_names))
    # out_classes
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(list(text_origin), label, fill=(0, 0, 0), font=font)
        del draw

    return image


# 12
def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


# 13
def get_test():
    # list images names And its annotation
    test_file = open("./coco_test.txt", "r")
    lines = test_file.readlines()
    images = []
    labels = []
    for i in lines:
        images.append(i.split()[0])
        labels.append(i.split()[1:])
    return images, labels


# 14
def yolo_test(model_path="./yolocode/model_data/yolo_weights.h5"):
    """ This function read images from test set and make prediction it return ground truth and prediction """
    TESTED = 5000

    test_images, test_labels = get_test()
    # ['5,35,475,379,48', '0,287,342,426,lable']
    # load model
    yolo_model = load_model(model_path, compile=False)

    # get data
    anchors = get_anchors()
    class_names = get_classes()
    num_anchors = len(anchors)
    num_classes = len(class_names)

    ann = []
    det = []
    for i in range(len(test_images)):
        print("Testing : " + str(i) + "/" + str(TESTED), end="\r")
        img = test_images[i]
        labels = test_labels[i]

        # read image
        image = read_image(img)

        ### Create annotation formate
        for k in range(len(labels)):
            labels[k] = labels[k].split(",")
            labels[k] = [int(j) for j in labels[k]]  # create int list
        labels = np.array(labels)
        labels[:, [2, 1]] = labels[:, [1, 2]]
        labels = np.insert(labels, 0, 0, axis=1)

        labels = labels[:, [0, 5, 1, 2, 3, 4]]

        df = pd.DataFrame(labels, columns=['name', 'class', 'xmin', 'xmax', 'ymin', 'ymax'])
        df = df.replace({'name': 0}, img)
        df = df.replace({"class": dict(zip(range(len(class_names)), class_names))})
        df['xmin'] = df['xmin'].div(image.size[0])
        df['xmax'] = df['xmax'].div(image.size[0])
        df['ymin'] = df['ymin'].div(image.size[1])
        df['ymax'] = df['ymax'].div(image.size[1])
        ann.append(df)
        # preprocess image
        image_data = preprocess_image(image)  # (1, 416, 416, 3)

        yolo_outputs = yolo_model.predict(image_data)
        # (1, 13, 13, 255)     # (1, 26, 26, 255)      # (1, 52, 52, 255)

        out_boxes, out_scores, out_classes, _ = yolo_eval(yolo_outputs, anchors, num_classes,
                                                          [image.size[1], image.size[0]], max_boxes=20,
                                                          score_threshold=.6, iou_threshold=.5)
        # TODO: using labels and (out_boxes, out_classes) calcuate map

        # create output labels formate
        out_boxes = np.array(out_boxes)
        # correct boxes
        for j in range(len(out_boxes)):
            out_boxes[j][0] = max(0, out_boxes[j][0])
            out_boxes[j][1] = max(0, out_boxes[j][1])
            out_boxes[j][2] = min(image.size[1], out_boxes[j][2])
            out_boxes[j][3] = min(image.size[0], out_boxes[j][3])
        out_boxes[:, [2, 1]] = out_boxes[:, [1, 2]]  # swap xmin,xmax,ymin,ymax
        out_boxes = np.insert(out_boxes, 0, out_scores, axis=1)  # add score
        out_boxes = np.insert(out_boxes, 0, out_classes, axis=1)  # add class
        out_boxes = np.insert(out_boxes, 0, 0, axis=1)  # add 0 for img name

        df2 = pd.DataFrame(out_boxes, columns=['name', 'class', 'score', 'xmin', 'xmax', 'ymin', 'ymax'])
        df2 = df2.replace({'name': 0}, img)
        df2 = df2.replace({"class": dict(zip(range(len(class_names)), class_names))})
        df2['xmin'] = df2['xmin'].div(image.size[0])
        df2['xmax'] = df2['xmax'].div(image.size[0])
        df2['ymin'] = df2['ymin'].div(image.size[1])
        df2['ymax'] = df2['ymax'].div(image.size[1])
        det.append(df2)
        if i == TESTED:
            break
    ann = pd.concat(ann)
    det = pd.concat(det)
    return ann, det
