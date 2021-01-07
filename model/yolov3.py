"""
Description: YOLOv3 Model definition

Some parts of this file is based on YOLOv3 from Yunyang1994.
Reference: https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
"""

import tensorflow as tf
import numpy as np
from model.config import cfg

NUM_CLASS = cfg.YOLO.NUM_CLASSES
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


def bbox_iou(boxes1, boxes2):
    # Area = actual width * actual height
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # shape of [...,4(x1,y1,x2,y2)]
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # left_up = [max_x1,max_y1]
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    # right_down = [min_x2,min_y2]
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # get[width,height] of inter section, max be more or equal than 0
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # enclose_are -> The min box which could cover both box1 and box2
    # left_up->(min_x1,min_y1)
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    # right_down ->(max_x2,max_y2)
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    """

    :param pred: Decoded output, shape of [n,output_size,output_size,3,4+1+NUM_CLASS]
    :param conv: Raw output, shape of [n,output_size,output_size, (3 * (5+NUM_CLASS))]
    :param label: [n,output_size,output_size,3,5+NUM_CLASS]
    :param bboxes: [n,150,4(x,y,w,h)]
    :param i: type of the output size
    :return:
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # Conf and prob before sigmoid
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    # Positive or Negative
    respond_bbox = label[:, :, :, :, 4:5]
    # Prob for each class
    label_prob = label[:, :, :, :, 5:]

    # Shape of (n,output_size,output_size,3,1) calculate giou for each pair of bbox in the feature map
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    # Bigger the size of bbox, smaller bbox_loss_scale. To balance influence to loss from size
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # Bbox regression loss
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # Shape of param1:(n,output_size,output_size,3,1,4) param2:(n,1,1,1,150,4). Output shape of
    # (n,output_size,output_size,3,150) Calculate iou for each pred bbox to all 150 gt bbox
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Shape of (n,output_size,output_size,3,1) The max iou for each pred bbox
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    # Shape of (n,output_size,output_size,3,1) value of is_negative && max_iou<Threshold
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    # Loss of confidence
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate positive grid and negative grid with max_iou<Threshold lost
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
