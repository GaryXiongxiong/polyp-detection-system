"""
Description: Predicting method

Some parts of this file is based on YOLOv3 from Yunyang1994.
Reference: https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
"""

import tensorflow as tf
import numpy as np
import time
from model.config import cfg
import utils.util as util


def predict(img, model, nms_iou=0.45, conf_thresh=0.45):
    num_classes = cfg.YOLO.NUM_CLASSES
    input_size = cfg.YOLO.INPUT_SIZE
    frame_size = img.shape[:2]
    img_data = util.image_preporcess(img.copy(), [input_size, input_size])
    img_data = img_data[np.newaxis, ...].astype(np.float32)
    prev_time = time.time()
    pred_bbox = model.predict_on_batch(img_data)[1:6:2]
    curr_time = time.time()
    exec_time = curr_time - prev_time
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = util.postprocess_boxes(pred_bbox, frame_size, input_size, conf_thresh)
    bboxes = util.nms(bboxes, nms_iou, method='nms')
    return bboxes, exec_time
