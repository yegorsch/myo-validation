from Tkinter import *
from collections import deque
from threading import Lock
import time
import myo
import numpy as np
import scipy.signal as sp
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class EmgCollector(myo.DeviceListener):

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))




def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        t -= 1
    print('Recording!\n')

def generate_filename(name, points_num):
    return name + "_" + str(points_num) + ".csv"


def record(points_number, hub, filename="name.csv"):
    listener = EmgCollector(points_number)
    with hub.run_in_background(listener.on_event):
        while True:
            if len(listener.get_emg_data()) == points_number:
                hub.stop()
                emg_data = listener.get_emg_data()
                emg_data = np.array([x[1] for x in emg_data])
                np.savetxt(filename, emg_data, delimiter=",")
                print("Recorded successfully")
                return emg_data
def filter(emg):
    N = len(emg)
    wind_size = 40
    i_start = range(1, N - wind_size)
    i_stop = range(wind_size, N)
    EMG_av = np.zeros((N - wind_size, 8))
    for i in range(N - 5 - wind_size):
        sample = np.mean(emg[i_start[i]:i_stop[i], :], axis=0)
        EMG_av[i, :] = sample
    return EMG_av

def preprocess(emg):
    emg = np.abs(emg)
    filtered_emg = filter(emg)
    return filtered_emg

def preprocess_recordings(recorded_data, labels):
    y_trains = {}
    for gesture_name, gesture_data in recorded_data.items():
        recorded_data[gesture_name] = preprocess(gesture_data)
        y_trains[gesture_name] = np.array([labels[gesture_name]] * len(recorded_data)) # assigning y-labels to a gesture for LDA



def process_data(recorded_data):
    preprocess_recordings(recorded_data)


def main():
    myo.init(sdk_path='/Users/egor/Documents/University/myo_sdk')
    hub = myo.Hub()

    options = "1. Record gesture \n2. Process and exit \n3. Exit \n ->"

    recorded_data = {}
    labels = {}
    y = 0
    while True:
        choice = int(raw_input(options))
        if choice == 1:
            points_number = int(raw_input("Please enter number of points to record \n ->"))
            gesture_name = str(raw_input("Please enter gesture name \n ->"))
            filename = generate_filename(gesture_name, points_number)
            proceed = int(raw_input("File name is " + filename + " \n You will have 5 seconds to prepare. Ready? [1 - Yes/ 0 - No] \n ->"))
            if proceed == 1:
                countdown(5)
                emg_data = record(points_number, hub, filename=filename)
                recorded_data[gesture_name] = emg_data
                labels[gesture_name] = y
                y += 1
                plt.plot(emg_data)
                plt.show()
        elif choice == 2:
            process_data(recorded_data, labels)
            return
        else:
            return


if __name__ == '__main__':
    main()
