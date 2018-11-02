import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone

class Experiment:

    def __init__(self,
                 data,
                 labels,
                 lda=LDA(),
                 log_reg=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                 wind_size=30,
                 test_size=0.2):
        """
        :param lda: LDA model for training
        :param log_reg: Logistic regression model for training
        :param wind_size: Window size for filtering
        :param test_size: (0,1) value for test size
        :param data: {"palm_left":raw_numpy_array}
        :param labels: {"palm_left": int_class }
        """

        # Parameters
        self.lda = lda
        self.log_reg = log_reg
        self.cross_lda = clone(lda)
        self.log_reg = clone(log_reg)
        self.wind_size = wind_size
        self.test_size = test_size

        # Data
        self.data = data
        self.labels = labels
        self.label_name = {v: k for k, v in self.labels.iteritems()}
        self.train_data = None
        self.X_test = None
        self.y_test = None

        # Scores
        self.scores = {}  # Scores in like {score_name: score_val}

    def run(self):
        """
        Run the experiment with cross validation. Call scores() to get the results.
        """
        self.preprocess()  # Preprocess and fill test_data and train_data
        self.train()  # Fit the models
        self.cross_validate()
        self.train_test_validate()

    def cross_validate(self):
        scoresLda = cross_val_score(self.cross_lda, self.X_test, self.y_test, cv=None, scoring='accuracy')
        scoresLog = cross_val_score(self.log_reg, self.X_test, self.y_test, cv=None, scoring='accuracy')
        self.scores["models"] = {"lda": scoresLda.mean(), "log": scoresLog.mean()}

    def train_test_validate(self):
        log_prediction = self.log_reg.predict(self.X_test)
        gestures = {}
        self.scores["gestures"] = gestures
        for i in range(len(self.data)):
            occurance = 0.0
            for j in range(len(log_prediction)):
                if log_prediction[j] == self.y_test[j] and self.y_test[j] == i:
                    occurance += 1
            self.scores["gestures"]['log_' + self.label_name[i]] = 0 if (self.y_test == i).sum() == 0 else float(occurance / ((self.y_test == i).sum()))

        lda_prediction = self.lda.predict(self.X_test)

        for i in range(len(self.data)):
            occurance = 0.0
            for j in range(len(log_prediction)):
                if lda_prediction[j] == self.y_test[j] and self.y_test[j] == i:
                    occurance += 1
            self.scores["gestures"]['lda_' + self.label_name[i]] = 0 if (self.y_test == i).sum() == 0 else float(occurance / ((self.y_test == i).sum()))

    def preprocess(self):
        for key, emg in self.data.items():
            emg = np.abs(emg)
            filtered_emg = self.filter(emg)
            self.data[key] = filtered_emg

    def train(self):
        X = []
        Y = []
        for gesture_name, gesture_data in self.data.items():
            # Assigning labels
            y = np.array([self.labels[gesture_name]] * len(gesture_data))  # assigning y-labels to a gesture for LDA

            X.extend(gesture_data)
            Y.extend(y)

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size, random_state=3)

        self.X_test = X_test
        self.y_test = y_test

        self.lda.fit(X_train, y_train)
        self.log_reg.fit(X_train, y_train)

    # def filter(self, emg):
    #     print(len(emg))
    #     N = len(emg)
    #     wind_size = self.wind_size
    #     i_start = range(1, N - wind_size)
    #     i_stop = range(wind_size, N)
    #     EMG_av = np.zeros((N, 8))
    #     for i in range(N - 5 - wind_size):
    #         sample = np.mean(emg[i_start[i]:i_stop[i], :], axis=0)
    #         EMG_av[i, :] = sample
    #     return EMG_av


    def filter(self, emg):
        rows = int(len(emg) / self.wind_size)
        cols = 8
        result = np.zeros((rows, cols))
        for i in range(8):
            channel = np.array_split(emg[:, i], rows)
            result[:, i] = [np.mean(l) for l in channel]
        return result