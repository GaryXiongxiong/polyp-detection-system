import sys
import PySimpleGUI as sg
import tensorflow as tf
import threading
import os

from model.config import cfg
from interface.detect import realtime_detect_ui
from interface.detect_data import detect_on_dataset_ui
from interface.annotation import annotation_ui
from interface.train_on_data import train_on_data_ui
from interface.benchmark import benchmark_ui
from utils.WindowDecorator import Window


def main_window():
    global model
    global load_thread
    global save_thread
    global model_path
    global amended
    sg.theme(cfg.UI.THEME)

    menu_def = [['File', ['Open', '!Save', '!Save as..', 'Exit']],
                ['Model', ['!Realtime Detection', '!Detect on dataset', '!Benchmark', '!Train on Data']],
                ['Data', ['Annotate']]]

    menu_def_activated = [['File', ['!Open', 'Save', 'Save as..', 'Exit']],
                          ['Model', ['Realtime Detection', 'Detect on dataset', 'Benchmark', 'Train on Data']],
                          ['Data', ['Annotate']]]

    main_layout = [
        [sg.Menu(menu_def, key='-MENU-')],
        [sg.Text('Polyp Detection System', pad=(20, 20), font=(cfg.UI.FONT, 25), justification='center')],
        [sg.Text('No model loaded', font=(cfg.UI.FONT, 18), pad=(20, 20), size=(50, 1), justification='center',
                 key="-MODEL-INFO-")],
        [sg.Button('Open Model', pad=(10, 10), size=(15, 1), key='-OPEN-'),
         sg.Button('Exit', pad=(10, 10), size=(15, 1))]]

    window = Window('Polyp Detect System', main_layout, element_justification='center', font=(cfg.UI.FONT, 16))

    def on_close():
        window.disable()
        global amended
        if amended:
            confirmation = sg.popup_yes_no('Do you want to exit without saving your trained model?', title="Exit..")
            if confirmation in ['No', None]:
                window.enable()
                return
        if window.CurrentlyRunningMainloop:
            amended = False
            try:
                window.TKroot.quit()
                window.TKroot.destroy()
                window.RootNeedsDestroying = True
                window.TKrootDestroyed = True
            finally:
                sys.exit()

    window.read(timeout=0)
    window.TKroot.protocol('WM_DELETE_WINDOW', on_close)
    while True:
        event, values = window.read(timeout=100)
        # Exit
        if event in ['Exit']:
            if amended:
                confirm = sg.popup_yes_no('Do you want to exit without saving your trained model?', title="Exit..")
                if confirm in ['No', None]:
                    continue
            break

        elif event in ('Open', '-OPEN-'):
            window.disable()
            input_model_path = sg.popup_get_folder('Please select the folder of model contains saved_model.pb',
                                                   title='Select folder of model', keep_on_top=True,
                                                   initial_folder='./')
            window.enable()
            window.bring_to_front()
            if input_model_path is not None:
                model_path = input_model_path
                model = None
                load_thread = threading.Thread(target=load_model, args=(model_path,), daemon=True)
                load_thread.start()

        elif event in ['Annotate']:
            window.disable()
            annotation_ui()
            window.enable()
            window.bring_to_front()

        if load_thread is not None:
            sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, 'Loading', time_between_frames=10)
            window.disable()
            if not load_thread.is_alive():
                # Update menu and button based on if model is loaded
                if model is not None:
                    window['-MODEL-INFO-'].update(
                        value='{} Model loaded, including {} layers.'.format(os.path.basename(model_path),
                                                                             len(model.layers)))
                    window['-MENU-'].update(menu_definition=menu_def_activated)
                else:
                    sg.popup_error('The selected model folder is invalid', title='Invalid Model')
                    window['-MODEL-INFO-'].update(value='No model loaded')
                    window['-MENU-'].update(menu_definition=menu_def)
                sg.popup_animated(None)
                window.enable()
                window.bring_to_front()
                load_thread = None

        if model is not None:
            if event == 'Realtime Detection':
                window.disable()
                realtime_detect_ui(model)
                window.enable()
                window.bring_to_front()
            elif event == 'Detect on dataset':
                window.disable()
                detect_on_dataset_ui(model)
                window.enable()
                window.bring_to_front()
            elif event == 'Benchmark':
                window.disable()
                benchmark_ui(model)
                window.enable()
                window.bring_to_front()
            elif event == 'Train on Data':
                window.disable()
                new_model, new_path, amended = train_on_data_ui(model, model_path)
                if new_model is not None:
                    model = new_model
                if new_path is not None:
                    model_path = new_path
                window.enable()
                window.bring_to_front()
            elif event == 'Save':
                if amended:
                    save_thread = threading.Thread(target=model.save, args=(model_path,))
                    save_thread.start()
            elif event == 'Save as..':
                save_as_path = sg.popup_get_folder("Select folder to save model", "Save As ..", initial_folder="./")
                if save_as_path is not None:
                    save_thread = threading.Thread(target=model.save, args=(save_as_path,))
                    save_thread.start()
                    model_path = save_as_path
                    window['-MODEL-INFO-'].update(
                        value='{} Model loaded, including {} layers.'.format(os.path.basename(model_path),
                                                                             len(model.layers)))

            if save_thread is not None:
                sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, 'Saving', time_between_frames=10)
                window.disable()
                if not save_thread.is_alive():
                    sg.popup_animated(None)
                    sg.popup_ok("Model saved")
                    window.enable()
                    window.bring_to_front()
                    save_thread = None
                    amended = False

    window.close()


def load_model(path):
    global model
    try:
        model = tf.keras.models.load_model(path)
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "batch_normalization_51":
                break
        model.summary()
    except Exception as e:
        print('invalid Model')
        print("{}".format(e))
        print(sys.exc_info()[0])


if __name__ == "__main__":
    os.environ['DISPLAY'] = "localhost:10.0"
    model = None
    load_thread = None
    save_thread = None
    amended = False
    model_path = None
    # Debug
    # sg.show_debugger_popout_window()
    main_window()
    sys.exit()
