from collections import deque
from threading import Lock
import csv
import pandas as pd
import os
import fnmatch
import time
import myo
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def record(points_number, myo, filename="name.csv"):
    hub = myo.Hub()
    listener = EmgCollector(points_number)
    listener.emg_data_queue.clear()
    while hub.run(listener.on_event, 500):
        print("Recorded points: " + str(len(listener.emg_data_queue)))
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

def training(recorded_data, labels):
    y_trains = {}
    test_data = {}
    X_train = []
    y_train = []
    for gesture_name, gesture_data in recorded_data.items():
        # X train
        processed_emg = preprocess(gesture_data)
        recorded_data[gesture_name] = processed_emg
        # y
        y = np.array([labels[gesture_name]] * len(processed_emg)) # assigning y-labels to a gesture for LDA
        y_trains[gesture_name] = y
        # splitting x and y into train and test set
        X_train_gesture, X_test_gesture, y_train_gesture, y_test_gesture = train_test_split(processed_emg, y, test_size=0.2, random_state=0)
        X_train.extend(X_train_gesture)
        y_train.extend(y_train_gesture)
        test_data[gesture_name] = (X_test_gesture, y_test_gesture)
    # Models
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    lda = LDA()
    logistic_regression = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial') if len(recorded_data.items()) >= 2 else lda
    lda.fit(X_train, y_train)
    logistic_regression.fit(X_train, y_train)

    return test_data, logistic_regression, lda

def process_recordings(test_data,logistic_regression,lda):

    #Cross validation
    scores_lda = []
    scores_log = []
    names_lda = []
    names_log = []
    for key, value in test_data.items():
        log_score = logistic_regression.score(value[0], value[1])
        lda_score = lda.score(value[0], value[1])
        scores_log.append(log_score)
        scores_lda.append(lda_score)
        names_lda.append(key + "\n%.3f" % lda_score)
        names_log.append(key + "\n%.3f" % log_score)
    # Plotting
    y_pos = np.arange(len(names_log))
    plt.bar(y_pos, scores_lda, align='center', alpha=0.5)
    plt.xticks(y_pos, names_lda)
    plt.ylabel('Score')
    plt.title('Gesture LDA')
    plt.show()

    plt.bar(y_pos, scores_log, align='center', alpha=0.5)
    plt.xticks(y_pos, names_log)
    plt.ylabel('Score')
    plt.title('Gesture Logistic Regression')
    plt.show()

    # Total performance among LDA and Logistic
    avg_algvals = []
    avg_names = []
    avg_algvals.append(sum(scores_lda) / float(len(scores_lda)))
    avg_algvals.append(sum(scores_log) / float(len(scores_log)))
    avg_names.append("LDA \n%.5f" % avg_algvals[0])
    avg_names.append("LOG \n%.5f" % avg_algvals[1])

    plt.bar([1,2], avg_algvals, align='center', alpha=0.5)
    plt.xticks([1,2], avg_names)
    plt.ylabel('Score')
    plt.title('Gesture Logistic Regression')
    plt.show()

def createModel(model_name,points_number):
    logname = model_name + "_logistic.sav"
    ldaname = model_name + "_lda.sav"
    read_data, readLabels = parseFiles(points_number)
    res2 = training(read_data, readLabels)
    pickle.dump(res2[1], open(logname, 'wb'), protocol=2)
    pickle.dump(res2[2], open(ldaname, 'wb'), protocol=2)

def parseFiles(points_number):
    listOfFiles = os.listdir('.')
    pattern = "*.csv"
    k = 0
    readLabels = {}
    read_data = {}
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            gest_name = entry.split("_")[0]
            dataset = pd.read_csv(entry)
            readLabels[gest_name] = k
            k += 1
            read_data[gest_name] = dataset.values[:points_number]
            # print(dataset.values[:points_number])
    return read_data, readLabels

def main():
    myo.init(sdk_path='C:/Users/Dinmukhammed/Desktop/MYO/myo-sdk-win-0.9.0')

    options = "\n1. Record gesture \n2. Process and exit \n3. Show and train prerecorded data. \n4. Save models. \n5. Exit. \n->"

    recorded_data = {}
    labels = {}
    y = 0
    points_number = 1000
    while True:
        choice = int(input(options))
        if choice == 1:
            gesture_name = str(input("Please enter gesture name \n ->"))
            filename = generate_filename(gesture_name, points_number)
            proceed = int(input("File name is " + filename + " \n You will have 3 seconds to prepare. Ready? [1 - Yes/ 0 - No] \n ->"))
            if proceed == 1:
                countdown(3)
                emg_data = record(points_number, myo, filename=filename)
                recorded_data[gesture_name] = emg_data
                labels[gesture_name] = y
                y += 1
                plt.plot(emg_data)
                plt.show()
        elif choice == 2:
            res = training(recorded_data, labels)
            process_recordings(res[0], res[1], res[2])

        elif choice == 3:
            res = parseFiles(points_number)
            res2 = training(res[0], res[1])
            process_recordings(res2[0], res2[1], res2[2])

        elif choice == 4:
            model_name = str(input("Please enter model name \n ->"))
            createModel(model_name, points_number)
        else:
            return


if __name__ == '__main__':
    main()
