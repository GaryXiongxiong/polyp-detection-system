import PySimpleGUI as sg
import cv2
import utils.util as util
import skimage.io as io
import numpy as np
import os
from model.config import cfg
from utils.data import placeholder
from utils.WindowDecorator import Window

amended = False


def annotation_ui():
    global amended
    sg.theme(cfg.UI.THEME)

    col1 = [
        [sg.Button('Open', size=(10, 1)), sg.Button('Save', size=(10, 1), disabled=True),
         sg.Button('Save as..', size=(10, 1), disabled=True),
         sg.Button('Exit', size=(10, 1))],
        [sg.Image(data=placeholder, key='-SCREEN-')]
    ]
    col2 = [
        [sg.Text('Images')],
        [sg.Listbox([], size=(20, 15), key='-IMAGE-', enable_events=True)],
        [sg.Text('Objects')],
        [sg.Listbox([], size=(20, 5), key='-OBJECT-', enable_events=True)],
        [sg.Button('Add', size=(10, 1), disabled=True), sg.Button('Delete', size=(10, 1), disabled=True)]
    ]

    layout = [
        [sg.Column(col1), sg.Column(col2, pad=(0, 10))]
    ]

    window = Window('Annotation', layout=layout, font=(cfg.UI.FONT, 12))

    anno_data = None
    anno_path = None

    def on_close():
        global amended
        if amended:
            confirm = sg.popup_yes_no('Do you want to exit without saving your amendment?', title="Exit..")
            if confirm in ['No', None]:
                return
        if window.CurrentlyRunningMainloop:
            amended = False
            window.TKroot.quit()
            window.TKroot.destroy()
            window.RootNeedsDestroying = True
            window.TKrootDestroyed = True

    window.finalize()
    window.TKroot.protocol('WM_DELETE_WINDOW', on_close)

    while True:
        event, value = window.Read()

        if event in ['Exit', None]:
            if amended:
                confirm = sg.popup_yes_no('Do you want to exit without saving your amendment?', title="Exit..")
                if confirm in ['No', None]:
                    continue
            break

        # Load annotation file
        if event in ['Open']:
            annotation_file = None
            anno_path = sg.popup_get_file('Annotation file', 'Open the annotation file:',
                                          file_types=(('Annotation file', 'annotation.txt'), ('All files', '*.*')),
                                          initial_folder='./')
            if anno_path is None:
                continue
            dir_path = os.path.dirname(anno_path)
            try:
                annotation_file = open(anno_path, 'r+')
                annotations = annotation_file.readlines()
                anno_data = []
                for anno in annotations:
                    anno = anno.strip()
                    bboxes, img_path = util.parse_annotation(anno)
                    img_path = os.path.join(dir_path, img_path)
                    anno_data.append([img_path, bboxes])
                if len(anno_data) == 0:
                    raise IOError()
                anno_data = np.array(anno_data)
                window['-IMAGE-'].update(values=[os.path.basename(x) for x in anno_data[:, 0]], set_to_index=0)
                window['-OBJECT-'].update(values=range(1, len(anno_data[0][1]) + 1))
                if len(anno_data[0][1]) > 0:
                    window['-OBJECT-'].update(set_to_index=0)
                amended = False
            except IOError:
                sg.PopupError('Invalid Annotation')
                anno_data = None
            except ValueError:
                sg.PopupError('Invalid Annotation format')
                anno_data = None
            finally:
                if annotation_file is not None:
                    annotation_file.close()

        if anno_data is not None:
            if len(window['-IMAGE-'].GetIndexes()) > 0:
                image_idx = window['-IMAGE-'].GetIndexes()[0]
            else:
                continue
            if event in ['-IMAGE-']:
                window['-OBJECT-'].update(values=range(1, len(anno_data[image_idx][1]) + 1))
                if len(anno_data[image_idx][1]) > 0:
                    window['-OBJECT-'].update(set_to_index=0)

            # Check number of objects in this image
            if len(window['-OBJECT-'].GetIndexes()) > 0:
                object_idx = window['-OBJECT-'].GetIndexes()[0]
            else:
                object_idx = -1
                window['Delete'].update(disabled=True)

            # Load image
            if anno_data[image_idx][0].endswith('.tif'):
                img_RGB = io.imread(anno_data[image_idx][0])
                img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(anno_data[image_idx][0])

            # Add button
            if event in ['Add']:
                img_to_draw = util.draw_bbox(img.copy(), anno_data[image_idx][1])
                input_coor = util.get_bbox_from_UI(img_to_draw)
                if input_coor is not None:
                    anno_data[image_idx][1].append([input_coor[0], input_coor[1], input_coor[2], input_coor[3], 1.0, 0])
                    window['-OBJECT-'].update(values=range(1, len(anno_data[image_idx][1]) + 1))

            # Delete button
            if event in ['Delete']:
                anno_data[image_idx][1].pop(object_idx)
                window['-OBJECT-'].update(values=range(1, len(anno_data[image_idx][1]) + 1))

            # Check number of objects in this image again after amending
            if event in ['Add', 'Delete']:
                amended = True
                if len(window['-OBJECT-'].GetIndexes()) > 0:
                    object_idx = window['-OBJECT-'].GetIndexes()[0]
                else:
                    object_idx = -1
                    window['Delete'].update(disabled=True)
            # Save and Save as
            if event in ['Save']:
                util.save_anno(anno_data, anno_path)
                amended = False
                sg.popup_ok('Annotation saved', title='Annotation saved')
            if event in ['Save as..']:
                save_as_path = sg.popup_get_file('Select file path to save annotation', 'Save as..',
                                                 default_extension='txt', save_as=True,
                                                 file_types=(
                                                     ('Annotation file', 'annotation.txt'), ('All files', '*.*')))
                if save_as_path is not None:
                    util.save_anno(anno_data, save_as_path, copy_image=True)
                    anno_path = save_as_path
                    amended = False
                    sg.popup_ok('Annotation saved at ' + save_as_path, title='Annotation saved')
            # Display image and bounding boxes
            draw_img = util.draw_bbox(img.copy(), anno_data[image_idx][1], highlight=object_idx)
            img_reshaped = draw_img
            if draw_img.shape[1] > 800:
                img_reshaped = cv2.resize(draw_img, (800, 800 * img.shape[0] // img.shape[1],))
            img_bytes = cv2.imencode('.png', img_reshaped)[1].tobytes()
            window['-SCREEN-'].update(data=img_bytes)

            window['Save'].update(disabled=not amended)
            window['Save as..'].update(disabled=anno_data is None)
            window['Add'].update(disabled=anno_data is None)
            window['Delete'].update(disabled=anno_data is None)

    window.close()

    del window
    del layout
    del col1
    del col2
