import PySimpleGUI as sg
import cv2
from model.predict import predict
import utils.util as util
import time
import os
from model.config import cfg
from utils.WindowDecorator import Window


def realtime_detect_ui(model):
    sg.theme(cfg.UI.THEME)
    layout = [
        [sg.Image('', key='-SCREEN-')],
        [sg.Text('', size=(30, 1), key='-PROCESS-TIME-')],
        [sg.Text('Confidence Threshold'), sg.Slider((0, 1), 0.4, 0.05, orientation='h', key='-CONF-SLIDER-'),
         sg.Text('NMS IOU Threshold'), sg.Slider((0, 1), 0.45, 0.05, orientation='h', key='-NMS-SLIDER-')],
        [sg.Button('Record', size=(10, 1), key='-REC-'), sg.Button('Collect', size=(10, 1), key='-COL-'),
         sg.Button('Close', size=(10, 1))]
    ]
    cam_no = sg.popup_get_text('Detection device number:', default_text='0')
    if cam_no is None:
        return

    # Initialise resource variables
    cam = cv2.VideoCapture()
    window = Window('Realtime Detect', layout=layout, finalize=True, font=cfg.UI.FONT)
    record = False
    collect = False
    anno_file = None
    collect_path = None
    out = None
    anno_index = 1

    try:
        cam_no = int(cam_no)
        cam.open(cam_no)
        while True:
            event, value = window.read(timeout=0)
            flag, img = cam.read()
            if event in [None, 'Close']:
                break
            # Record button pushed
            if event in ['-REC-']:
                record = not record
                window['-REC-'].update(text=('Stop' if record else 'Record'))
            # Collect button pushed
            if event in ['-COL-']:
                if collect:
                    anno_file.flush()
                collect = not collect
                window['-COL-'].update(text=('Stop' if collect else 'Collect'))

            # Read from Webcam successfully
            if flag:
                # Predict bboxes
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bboxes, exe_time = predict(img_RGB, model, value['-NMS-SLIDER-'], value['-CONF-SLIDER-'])
                # Draw bbox to img
                pred_img = util.draw_bbox(img.copy(), bboxes)

                # Record video to the file
                if record:
                    if out is None:
                        out = cv2.VideoWriter(time.strftime('%Y%m%d-%H%M', time.localtime(time.time())) + '.mp4', -1,
                                              20.0, (640, 480))
                    out.write(pred_img)

                # Collect image and annotation
                if collect:
                    if collect_path is None:
                        collect_path = './anno_data/collected_data_' + time.strftime('%Y%m%d-%H%M',
                                                                                     time.localtime(time.time()))
                        os.mkdir(collect_path)
                        os.mkdir(os.path.join(collect_path, 'img'))
                    if anno_file is None:
                        anno_file = open(os.path.join(collect_path, 'annotation.txt'), 'a')
                        anno_index = 1
                    img_path = os.path.join('img', str(anno_index) + '.jpg')
                    cv2.imwrite(os.path.join(collect_path, img_path), img)
                    anno_line = util.encode_annotation(bboxes, img_path)
                    anno_file.write(anno_line.strip() + '\n')
                    anno_index += 1
                # Display predicted image
                img_bytes = cv2.imencode('.png', pred_img)[1].tobytes()
                window['-SCREEN-'].update(data=img_bytes)
                window['-PROCESS-TIME-'].update(value='Process time: {}ms'.format(int(exe_time * 1000)))
            else:
                raise ValueError

    except ValueError:
        sg.popup_error("Selected Device is unavailable")
        return
    finally:
        cam.release()
        if out is not None:
            out.release()
        if anno_file is not None:
            anno_file.close()
        window.close()
        del cam
        del window
        del layout
