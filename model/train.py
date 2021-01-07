"""
Description: Model training class

Some parts of this file is based on YOLOv3 from Yunyang1994.
Reference: https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
"""

import tensorflow as tf
from model.yolov3 import compute_loss


class ModelTrainer(object):

    def __init__(self, model, dataset, lr, epoch, mess_queue):
        self.mess_queue = mess_queue
        self.mess_queue.put((0, "Initialising..."))
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.epoch = epoch
        self.step_per_epoch = len(dataset)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.total_step = self.step_per_epoch * self.epoch
        self.mess_queue.put((0, "Initialised"))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.need_stop = False

    def train_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.mess_queue.put((int(100 * self.global_steps / self.total_step),
                                 "=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                                 "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, self.optimizer.lr.numpy(),
                                                                           giou_loss, conf_loss,
                                                                           prob_loss, total_loss)))
            # update learning rate
            self.global_steps.assign_add(1)

    def train(self):
        for epoch in range(self.epoch):
            if self.need_stop:
                self.need_stop = False
                break
            self.mess_queue.put((int(100 * self.global_steps / self.total_step), "---Epoch{}---".format(epoch+1)))
            for image_data, target in self.dataset:
                if self.need_stop:
                    break
                self.train_step(image_data, target)

    def terminate(self):
        self.need_stop = True
