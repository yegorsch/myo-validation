import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Experiment:

    def __init__(self,
                 data,
                 labels,
                 lda=LDA(),
                 log_reg=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                 wind_size=40,
                 test_size=0.2):
        """
        :param lda: LDA model for training
        :param log_reg: Logistic regression model for training
        :param wind_size: Filtering window constant
        :param test_size: (0,1) value for test set size
        :param data: e.g. {"palm_left": raw_numpy_array}
        :param labels: e.g.  {"palm_left": 0 }
        """

        # Parameters
        self.lda = lda
        self.log_reg = log_reg
        self.wind_size=wind_size
        self.test_size = test_size

        # Data
        self.data = data
        self.labels = labels
        self.train_data = {}
        self.test_data = {}

        # Scores
        self.scores = {} #Scores in like {score_name: score_val}




    def run(self):
        """
        Run the experiment with cross validation. Call scores() to get the results.
        """
        self.preprocess()  # Preprocess and fill test_data and train_data
        self.train()  # Fit the models
        self.cross_validate()  # Cross vali

    def scores(self):
        return self.scores()

    def cross_validate(self):
        # Cross validation
        pass

    def preprocess(self):
        for key, emg in self.data.items():
            emg = np.abs(emg)
            filtered_emg = self.filter(emg)
            self.data[key] = filtered_emg

    def train(self, shuffle=False):
        y_trains = {}
        test_data = {}
        X_train = []
        y_train = []
        for gesture_name, gesture_data in self.data.items():
            # Assigning labels
            y = np.array([self.labels[gesture_name]] * len(gesture_data))  # assigning y-labels to a gesture for LDA
            y_trains[gesture_name] = y
            # splitting x and y into train and test set
            X_train_gesture, X_test_gesture, y_train_gesture, y_test_gesture = train_test_split(gesture_data, y,
                                                                                                test_size=self.test_size,
                                                                                                random_state=0)
            X_train.extend(X_train_gesture)
            y_train.extend(y_train_gesture)
            test_data[gesture_name] = (X_test_gesture, y_test_gesture)
        # Models
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if shuffle:
            shuffled_data = np.column_stack((X_train, y_train))
            np.random.shuffle(shuffled_data)
            y_train = shuffled_data[:, -1].copy()
            X_train = np.delete(shuffled_data, -1, axis=1)

        self.lda.fit(X_train, y_train)
        self.log_reg.fit(X_train, y_train)

    def filter(self, emg):
        N = len(emg)
        wind_size = self.wind_size
        i_start = range(1, N - wind_size)
        i_stop = range(wind_size, N)
        EMG_av = np.zeros((N - wind_size, 8))
        for i in range(N - 5 - wind_size):
            sample = np.mean(emg[i_start[i]:i_stop[i], :], axis=0)
            EMG_av[i, :] = sample
        return EMG_av
