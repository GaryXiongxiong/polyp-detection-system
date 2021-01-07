import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.predict import predict
import utils.util as util
import skimage.io as io
import os
from model.config import cfg
from utils.WindowDecorator import Window


def benchmark_ui(model):
    sg.theme(cfg.UI.THEME)
    layout = [
        [sg.Text('Ground truth annotation Path:')],
        [sg.Input(key='-INPUT-PATH-', enable_events=True),
         sg.FileBrowse(initial_folder='./', file_types=(('Annotation file', 'annotation.txt'), ('All files', '*.*')))],
        [sg.Text('Confidence Threshold'), sg.Slider((0, 1), 0.4, 0.05, orientation='h', key='-CONF-SLIDER-'),
         sg.Text('NMS IOU Threshold'), sg.Slider((0, 1), 0.45, 0.05, orientation='h', key='-NMS-SLIDER-')],
        [sg.B('Run Benchmark'), sg.B('Exit')]
    ]
    window = Window('Benchmark', layout, font=(cfg.UI.FONT, 12))
    while True:
        event, value = window.read()

        if event in ['Exit', None]:
            break
        # Start detection
        if event in ['Run Benchmark']:
            anno_path = value['-INPUT-PATH-']
            nms_iou = value['-NMS-SLIDER-']
            conf_thresh = value['-CONF-SLIDER-']
            anno_file = None
            tp = tn = fp = fn = 0
            try:
                # Open and parse annotation file
                anno_file = open(anno_path, 'r')
                dir_path = os.path.dirname(anno_path)
                annotations = anno_file.readlines()
                anno_data = []
                for anno in annotations:
                    anno = anno.strip()
                    bboxes, img_path = util.parse_annotation(anno)
                    img_path = os.path.join(dir_path, img_path)
                    anno_data.append([img_path, bboxes])
                if len(anno_data) == 0:
                    raise IOError()
                max_progress = len(anno_data)
                cur_progress = 0
                window.disable()
                sg.one_line_progress_meter('Calculating Benchmark', cur_progress, max_progress, key='-PROGRESS-')
                # Start to predict and assess:
                for entry in anno_data:
                    img_RGB = io.imread(entry[0])
                    cur_tp = cur_tn = cur_fp = cur_fn = 0
                    prediction, _ = predict(img_RGB, model, nms_iou, conf_thresh)
                    target_bboxes = entry[1]
                    hit_bboxes = []
                    if len(prediction) == 0 and len(target_bboxes) == 0:
                        cur_tn += 1
                    for bbox in prediction:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        centroid = ((coor[0] + coor[2]) / 2, (coor[1] + coor[3]) / 2)
                        hit = False
                        for t_bbox in target_bboxes:
                            if (t_bbox[0] <= centroid[0] <= t_bbox[2]) and (t_bbox[1] <= centroid[1] <= t_bbox[3]):
                                cur_tp += 1
                                hit = True
                                target_bboxes.remove(t_bbox)
                                hit_bboxes.append(t_bbox)
                                break
                        if not hit:
                            for t_bbox in hit_bboxes:
                                if (t_bbox[0] <= centroid[0] <= t_bbox[2]) and (t_bbox[1] <= centroid[1] <= t_bbox[3]):
                                    hit = True
                                    break
                        if not hit:
                            cur_fp += 1
                    cur_fn += len(target_bboxes)
                    tp += cur_tp
                    tn += cur_tn
                    fp += cur_fp
                    fn += cur_fn
                    cur_progress += 1
                    # Update progress bar
                    if not sg.one_line_progress_meter('Calculating Benchmark', cur_progress, max_progress,
                                                      key='-PROGRESS-'):
                        break

                if sg.popup_ok("Detection finished,press 'OK' to see the result", title="Finish") is not None:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].pie((tp, tn, fp, fn),
                               labels=["TP", "TN", "FP", "FN"],
                               autopct='%1.1f%%')
                    axs[0].axis('equal')
                    pre = tp / (tp + fp) if tp + fp != 0 else 0
                    rec = tp / (tp + fn) if tp + fn != 0 else 0
                    spe = tn / (fp + tn) if fp + tn != 0 else 0
                    f1 = (2 * pre * rec) / (pre + rec)
                    f2 = (5 * pre * rec) / (4 * pre + rec)
                    rects = axs[1].bar(('Prec', 'Rec', 'Spec', 'F1', 'F2'), (pre, rec, spe, f1, f2))
                    for rect in rects:
                        height = rect.get_height()
                        axs[1].annotate('{:.2f}%'.format(height * 100),
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    fig.suptitle('Benchmarks')
                    benchmark_img = os.path.join(dir_path, 'benchmark.png')
                    plt.savefig(benchmark_img)
                    # sg.popup_animated(benchmark_img, no_titlebar=False)
                    Window("Benchmark", [[sg.Image(benchmark_img)]]).read(close=True)

            except IOError as e:
                sg.one_line_progress_meter('Calculating Benchmark', 1, 1, key='-PROGRESS-')
                sg.PopupError("Invalid input file,\n{}".format(e))
            finally:
                if anno_file is not None:
                    anno_file.close()
                window.enable()

    window.close()
    del window
    del layout
