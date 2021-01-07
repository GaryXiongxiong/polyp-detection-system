import PySimpleGUI as sg
import cv2
from model.predict import predict
import utils.util as util
import skimage.io as io
import os
from model.config import cfg
from utils.WindowDecorator import Window


def detect_on_dataset_ui(model):
    sg.theme(cfg.UI.THEME)
    layout = [
        [sg.Text('Dataset Folder Path:')],
        [sg.Input(key='-INPUT-PATH-', enable_events=True), sg.FolderBrowse(initial_folder='./')],
        [sg.Text('Prediction Output Path:')],
        [sg.Input(key='-OUTPUT-PATH-'), sg.FolderBrowse(initial_folder='./')],
        [sg.Text('Annotation Output Path')],
        [sg.Input(key='-ANNO-PATH-'), sg.FileSaveAs(initial_folder='./')],
        [sg.Text('Input File Type:')],
        [sg.Combo(('.jpg', '.png', '.bmp', '.tif', '.tiff', '.gif'), readonly=True, default_value='.jpg', size=(15, 1),
                  key='-FILE-TYPE-')],
        [sg.Text('Confidence Threshold'), sg.Slider((0, 1), 0.4, 0.05, orientation='h', key='-CONF-SLIDER-'),
         sg.Text('NMS IOU Threshold'), sg.Slider((0, 1), 0.45, 0.05, orientation='h', key='-NMS-SLIDER-')],
        [sg.B('Detect'), sg.B('Exit')]
    ]
    window = Window('Detect Dataset', layout, font=(cfg.UI.FONT, 12))
    while True:
        event, value = window.read()

        if event in ['Exit', None]:
            break
        # Update output path and annotation file path according to input path
        if event in ['-INPUT-PATH-']:
            window['-OUTPUT-PATH-'].update(os.path.join(value['-INPUT-PATH-'], 'Prediction'))
            window['-ANNO-PATH-'].update(os.path.join(value['-INPUT-PATH-'], 'annotation.txt'))
        # Start detection
        if event in ['Detect']:
            input_path = value['-INPUT-PATH-']
            output_path = value['-OUTPUT-PATH-']
            anno_path = value['-ANNO-PATH-']
            file_type = value['-FILE-TYPE-']
            nms_iou = value['-NMS-SLIDER-']
            conf_thresh = value['-CONF-SLIDER-']
            anno_file = None
            try:
                anno_file = open(anno_path, 'w')
                images = os.listdir(input_path)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                # Count file conform the given file type
                max_process = [x.endswith(file_type) for x in images].count(True)
                cur_process = 1
                window.disable()

                for image in images:
                    if image.endswith(file_type):
                        # If user push cancel button on the progress popup
                        if not sg.one_line_progress_meter('Generating Prediction', cur_process, max_process,
                                                          key='-PROGRESS-'):
                            break

                        img_RGB = io.imread(os.path.join(input_path, image))
                        img = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2BGR)
                        # Predict
                        bboxes, _ = predict(img_RGB, model, nms_iou, conf_thresh)
                        pred_img = util.draw_bbox(img.copy(), bboxes)
                        cv2.imwrite(os.path.join(output_path, image), pred_img)
                        anno_line = util.encode_annotation(bboxes, image)
                        anno_file.write(anno_line.strip() + '\n')
                        cur_process += 1

            except IOError as e:
                sg.PopupError("Invalid input file,\n{}".format(e))
                sg.one_line_progress_meter('Generating Prediction', 1, 1, key='-PROGRESS-')
            finally:
                if anno_file is not None:
                    anno_file.close()
                window.enable()

            sg.popup_ok("Detection finished", title="Finish")

    window.close()
    del window
    del layout
