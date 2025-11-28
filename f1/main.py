from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import vgamepad as vg 
import time
import math
import numpy as np

from drive import accelerate, brake, steer_left, steer_right, reset_controls
gamepad = vg.VX360Gamepad()


def cli():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', 
                                                    ecg=True, 
                                                    imu=True, 
                                                    ppg=True, 
                                                    eda=True, 
                                                    emg=True,
                                                    filtering=True,
                                                    emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    i=0
    avg=0
    buffer_size = 10
    while True:
        data = odh.get_data(2)
        if i == buffer_size:
            i=0
            print(np.floor(avg/buffer_size))
            print("-----------------------------")
            avg=0
        avg+=data[0]['imu'][0]
        i+=1
        
        print(data[0]['imu'][0])
        print("-----------------------------")
        
        if data[0]['imu'][0][0] <= -7:
            #pyautogui.press('right')
            #gamepad.right_trigger(255)
            #gamepad.update()

            accelerate()
        if data[0]['imu'][0][0] >= 7:
            #pyautogui.press('right')
            #gamepad.right_trigger(255)
            #gamepad.update()

            brake()

        if data[0]['imu'][0][2] >= 2:
            
            steer_right()
        if data[0]['imu'][0][2] <= -2:
            steer_left()
           
        else:
            time.sleep(0.1)
            reset_controls()
            


    
            
if __name__ == "__main__":
    cli()