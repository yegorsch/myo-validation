import sys

import pandas as pd
import os
import fnmatch
import time
import myo
import numpy as np
from matplotlib import pyplot as plt
from collector import EmgCollector
from experiment import Experiment


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


# def createModel(model_name,points_number):
#     logname = model_name + "_logistic.sav"
#     ldaname = model_name + "_lda.sav"
#     read_data, readLabels = parseFiles(points_number)
#     res2 = training(read_data, readLabels)
#     pickle.dump(res2[1], open(logname, 'wb'), protocol=2)
#     pickle.dump(res2[2], open(ldaname, 'wb'), protocol=2)

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
    return read_data, readLabels


def main():
    myo.init(sdk_path='/Users/egor/Documents/University/myo_sdk')

    options = "\n1. Record gesture \n2. Process and exit \n3. Show and train prerecorded data. \n4. Save models. \n5. Exit. \n->"

    recorded_data = {}
    labels = {}
    y = 0
    points_number = 1000
    while True:
        choice = int(input(options))
        if choice == 1:
            gesture_name = str(raw_input("Please enter gesture name \n ->"))
            filename = generate_filename(gesture_name, points_number)
            proceed = int(input(
                "File name is " + filename + " \n You will have 3 seconds to prepare. Ready? [1 - Yes/ 0 - No] \n ->"))
            if proceed == 1:
                countdown(3)
                emg_data = record(points_number, myo, filename=filename)
                recorded_data[gesture_name] = emg_data
                labels[gesture_name] = y
                y += 1
                plt.plot(emg_data)
                plt.show()
        elif choice == 2:
            exp = Experiment(data=recorded_data, labels=labels)
            exp.run()
            exp.scores
        elif choice == 3:
            res = parseFiles(points_number)
            plot_results(res)
        elif choice == 4:  # save model
            # model_name = str(input("Please enter model name \n ->"))
            # createModel(model_name, points_number)
            continue
        else:
            return


def plot_results(res):
    X = res[0]
    y = res[1]
    train_split_test(X, y)
    window_size_test(X, y)
    generate_graph_markdown()


def generate_graph_markdown():
    from os import listdir
    from os.path import isfile, join
    path = "./graphs"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    sample = "![graph](MMM)\n"
    with open("./graphs/README.md", "w") as f:
        for file in files:
            if ".png" in file:
                f.write(sample.replace("MMM", file))


def train_split_test(X, y):

    #
    #       BARS
    #

    plt.figure(figsize=(20, 60))
    train_split_vals = np.linspace(0.8, 0.97, 20)
    windows_size = 20
    for i, train_split in enumerate(train_split_vals):
        exp = Experiment(data=X, labels=y, test_size=train_split, wind_size=windows_size)
        try:
            exp.run()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Broke on %f" % train_split)
            break
        scores = exp.scores["gestures"]
        names = scores.keys()
        results = scores.values()
        plt.subplot(len(train_split_vals), 1, i + 1)
        plt.bar(range(len(results)), results, align='center', alpha=0.5)
        plt.xticks(range(len(names)), names)
        plt.ylabel('Score')
        plt.title('Test size %.3f' % train_split + " Avg: %f" % np.average(results) + "Window size: %d" % windows_size)
    plt.savefig('./graphs/diff_test_size_bars_windsize%d.png' % windows_size)
    plt.show()

    #
    #       GRAPH
    #

    finals = []
    for i, train_split in enumerate(train_split_vals):
        exp = Experiment(data=X, labels=y, test_size=train_split, wind_size=windows_size)
        try:
            exp.run()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Broke on %f" % train_split)
            break
        scores = exp.scores["gestures"]
        results = scores.values()
        finals.append(np.average(results))
    plt.plot(train_split_vals, finals)
    plt.ylabel("Score")
    plt.xlabel("Test split")
    plt.title("Window size: 30")
    plt.savefig('./graphs/relation_winsize30.png')
    plt.show()


def window_size_test(X, y):
    lda_res = []
    log_res = []
    x_window_sizes = []
    wind_sizes = np.linspace(1, 300, 100)
    for wind_size in wind_sizes:
        exp = Experiment(data=X, labels=y, wind_size=wind_size, test_size=0.5)
        try:
            exp.run()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break
        scores = exp.scores["models"]
        lda_res.append(scores["lda"])
        log_res.append(scores["log"])
        x_window_sizes.append(wind_size)
    plt.plot(x_window_sizes, lda_res)
    plt.ylim((min(lda_res), 1.1))
    plt.ylabel("LDA score")
    plt.xlabel("Window size")
    plt.savefig('./graphs/window_size_vs_ldascore.png')
    plt.show()
    plt.plot(x_window_sizes, log_res)
    plt.ylim((min(log_res) - 0.5, 1.1))
    plt.ylabel("Logistic regression score")
    plt.xlabel("Window size")
    plt.savefig('./graphs/window_size_vs_logscore.png')
    plt.show()


if __name__ == '__main__':
    main()
