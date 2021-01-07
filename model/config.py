"""
Description: Configuration file for this application

Some parts of this file is based on YOLOv3 from Yunyang1994.
Reference: https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
"""

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.YOLO = edict()

__C.YOLO.INPUT_SIZE = 416
__C.YOLO.NUM_CLASSES = 1
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.ANCHORS = './data/anchors/polyp_anchors.txt'
__C.YOLO.IOU_LOSS_THRESH = 0.6
__C.YOLO.DATA_IOU_THRESH = 0.3
__C.YOLO.CLASSES = './data/classes/polyp.names'

__C.UI = edict()
__C.UI.THEME = 'Reddit'
__C.UI.FONT = 'Calibri'
