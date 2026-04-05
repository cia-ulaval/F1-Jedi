try:
    from config import Config
except ImportError:
    from code_etienne.config import Config
import libemg
import numpy as np
import os
import time
import sys
import warnings
import datetime
import shutil

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")

config = Config()

def prepare_streamer(streaming=False):
    shared_memory_items = []
    shared_memory_items.append(["emg",       (config.EMG_FS,1), np.double])
    shared_memory_items.append(["emg_count", (1,1),    np.int32])

    streamer, smi = libemg.streamers.sifi_biopoint_streamer(
        name = "BioPoint_v1_3",
        shared_memory_items = shared_memory_items,
        ecg = False,
        emg = True, 
        eda = False,
        imu = False,
        ppg = False,
        filtering = True, 
        emg_notch_freq = config.NOTCH,
        streaming=streaming,
    )
    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
    return odh

def install_online_filters(odh:libemg.data_handler.OnlineDataHandler):
    fi = libemg.filtering.Filter(config.EMG_FS)
    fi.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    odh.install_filter(fi)

def ensure_folder():
    dataset_path = os.path.abspath(str(config.DATA_PATH_SUBJECT))
    # Ensure folder exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
    # Check if folder has any files/subfolders
    has_data = any(os.scandir(dataset_path))
    
    if has_data:
        print(f"Data found in '{dataset_path}'.")
        while True:
            choice = input("Choose action - [O]verwrite, [R]ename, [C]ancel: ").strip().lower()
            if choice == "o" or choice == "overwrite":
                confirm = input(f"Are you sure you want to DELETE all contents of '{dataset_path}'? Type 'yes' to confirm: ").strip().lower()
                if confirm == "yes":
                    shutil.rmtree(dataset_path)
                    os.makedirs(dataset_path, exist_ok=True)
                    print(f"Contents of '{dataset_path}' deleted.")
                    break
                else:
                    print("Overwrite cancelled.")
                    continue
            elif choice == "r" or choice == "rename":
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = f"{dataset_path}_{timestamp}"
                # Ensure unique name
                i = 1
                candidate = new_path
                while os.path.exists(candidate):
                    candidate = f"{new_path}_{i}"
                    i += 1
                os.makedirs(candidate, exist_ok=True)
                dataset_path = candidate
                print(f"Dataset directory renamed to '{dataset_path}'. Using that path for this run.")
                
                break
            elif choice == "c" or choice == "cancel":
                print("Operation cancelled. Exiting.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter O, R, or C.")
    else:
        # Ensure folder exists (redundant safe-guard)
        os.makedirs(dataset_path, exist_ok=True)
        
    return dataset_path

def collect_data():
    dataset_path = ensure_folder()
    print("Collecting data, data_folder: ", dataset_path)
    odh = prepare_streamer()
    args = {
        "online_data_handler": odh,
        "media_folder": config.MEDIA_FOLDER,
        "data_folder": dataset_path,
        "num_reps": config.NUM_REPS,
        "rep_time": config.REP_TIME,
        "rest_time": config.REST_TIME,
        "auto_advance": True
    }
    gui = libemg.gui.GUI(odh, args=args, debug=False, width=config.GUI_WIDTH, height=config.GUI_HEIGHT)
    # gui.clean_up_on_kill = True
    gui.download_gestures(config.CLASSES, config.MEDIA_FOLDER)
        
    gui.start_gui()
    
    print("Data collection complete, data saved to: ", dataset_path)


if __name__ == "__main__":
    collect_data()
