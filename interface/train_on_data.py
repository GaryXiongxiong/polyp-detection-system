import queue
from queue import Queue

import PySimpleGUI as sg
import time
import threading
from model.dataset import Dataset
from model.config import cfg
from model.train import ModelTrainer
from utils.WindowDecorator import Window

amended = False


def train_on_data_ui(model, model_path):
    save_path = model_path
    sg.theme(cfg.UI.THEME)

    layout = [
        [sg.Text("Epoch:"), sg.Spin(values=[x for x in range(100)], initial_value=5, key="-EPOCH-", size=(10, 1)),
         sg.Text("Learning Rate:"), sg.Input("1e-5", key="-LR-", size=(10, 1)),
         sg.Text("Batch Size"), sg.Spin(values=[x for x in range(64)], initial_value=2, key="-BS-", size=(10, 1))],
        [sg.Text("Data annotation file:"), sg.Input(size=(70, 1), key="-PATH-"),
         sg.FileBrowse(file_types=(("Annotation File", "annotation.txt"), ("ALL Files", "*.*")), initial_folder="./",
                       size=(9, 1), key="-BROWSE-")],
        [sg.ProgressBar(100, size=(80, 30), key="-PB-")],
        [sg.Multiline("--Waiting for training task--\n", autoscroll=True, disabled=True, size=(100, 20), key="-OUT-")],
        [sg.B("Start Train", key="-START-", size=(15, 1)),
         sg.B("Save Model", key="-SAVE-", size=(15, 1), disabled=True),
         sg.B("Save Model As..", key="-SAVE-AS-", size=(15, 1)), sg.B("Exit", size=(15, 1))]
    ]

    window = Window("Train on Data", layout=layout, font=(cfg.UI.FONT, 12))
    message_queue = Queue()
    train_thread = None
    global amended
    save_thread = None
    dataset = None
    trainer = None

    while True:
        event, value = window.read(timeout=100)
        if event in [None, "Exit"]:
            break

        if event in ["-START-"]:
            epoch = int(value["-EPOCH-"])
            data_path = value["-PATH-"]
            batch_size = int(value["-BS-"])
            lr = float(value["-LR-"])
            try:
                dataset = Dataset(data_path, batch_size)
            except Exception as e:
                sg.popup_error(e)
                continue
            trainer = ModelTrainer(model, dataset, lr, epoch, message_queue)
            train_thread = threading.Thread(target=train_task, args=(trainer, message_queue), daemon=True)
            amended = True
            train_thread.start()
            window["-START-"].update(disabled=True)
            window["-SAVE-"].update(disabled=True)
            window["-SAVE-AS-"].update(disabled=True)
            window["-EPOCH-"].update(disabled=True)
            window["-LR-"].update(disabled=True)
            window["-BS-"].update(disabled=True)
            window["-PATH-"].update(disabled=True)
            window["-BROWSE-"].update(disabled=True)

        if event in ["-SAVE-"]:
            if amended:
                save_thread = threading.Thread(target=model.save, args=(save_path,))
                save_thread.start()

        if event in ["-SAVE-AS-"]:
            save_as_path = sg.popup_get_folder("Select folder to save model", "Save As ..", initial_folder="./")
            if save_as_path is not None:
                save_thread = threading.Thread(target=model.save, args=(save_as_path,))
                save_thread.start()
                save_path = save_as_path

        if train_thread is not None:

            while True:
                try:
                    progress, message = message_queue.get_nowait()
                except queue.Empty:
                    break
                if message.startswith("ERROR:"):
                    ml_print_line(window["-OUT-"], message, color="red")
                else:
                    ml_print_line(window["-OUT-"], message)
                if progress is not None:
                    window["-PB-"].UpdateBar(progress)

            if not train_thread.is_alive():
                ml_print_line(window["-OUT-"], "Task Over", color="green")
                train_thread = None
                window["-START-"].update(disabled=False)
                window["-SAVE-"].update(disabled=False)
                window["-SAVE-AS-"].update(disabled=False)
                window["-EPOCH-"].update(disabled=False)
                window["-LR-"].update(disabled=False)
                window["-BS-"].update(disabled=False)
                window["-PATH-"].update(disabled=False)
                window["-BROWSE-"].update(disabled=False)
                dataset = None
                trainer = None

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

    if trainer is not None:
        trainer.terminate()
    window.close()
    return model, save_path, amended


def ml_print_line(el, value, with_date=True, color=None):
    if with_date:
        cur_time = time.strftime('[%y-%m-%d %H:%M:%S]', time.localtime(time.time()))
        el.update(value=cur_time + " " + value + "\n", append=True, text_color_for_value=color)
    else:
        el.update(value=" " + value + "\n", append=True, text_color_for_value=color)


def train_task(trainer, mess_queue):
    try:
        trainer.train()
    except Exception as e:
        mess_queue.put((0, "ERROR: {}".format(e)))
        mess_queue.put((0, "ERROR: Please check the annotation file"))
        raise e
