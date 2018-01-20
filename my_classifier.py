from numpy import *

from sklearn.tree import DecisionTreeClassifier

class BatchClassifier:


    def __init__(self, window_size = 100, max_models = 10):
        self.H = []
        self.windowSize = window_size
        self.maxModels = max_models
        self.preData = []
        self.lastModelIndex = 0

    def addModel(self, model):
        if (len(self.H) < self.maxModels):
            self.H.append(model)

        else:
            self.H[self.lastModelIndex] = model

        self.incrementLastModelIndex()

    def incrementLastModelIndex(self):
        if (self.lastModelIndex == self.maxModels - 1):
            self.lastModelIndex = 0
        else:
            self.lastModelIndex = self.lastModelIndex + 1

    def partial_fit(self, X, y = None, classes = None):

        # N.B.: The 'classes' option is not important for this classifier
        if len(X) + len(self.preData) < self.windowSize :
            self.preData = self.preData + (X)

        else:
            while (True):
                useFromX = self.windowSize - len(self.preData)
                useFromY = self.windowSize - len(self.preData)

                if len(self.preData) == 0:
                    xToUse = X[:useFromX]
                    yToUse = y[:useFromY]
                else:
                    xToUse = self.preData + (X[:useFromX])
                    yToUse = self.preData + (y[:useFromY])
                    self.preData = []

                X = X[useFromX:]
                y = y[useFromY:]

                h = DecisionTreeClassifier()
                h.fit(xToUse, yToUse)
                self.addModel(h)

                if(len(X) < self.windowSize):
                    self.preData = X
                    break

        return self

    def getMajority(self, predictions):
        zeros, ones = 0, 0
        for prediction in predictions:
            if prediction == 0:
                zeros += 1
            else :
                ones += 1
        if (zeros > ones):
            return 0
        else:
            return 1

    def predict(self, X):
        N,D = X.shape
        predictions = []
        for decisionTree in self.H:
            predictions.append(decisionTree.predict(X))

        majority = self.getMajority(predictions)
        predictions = []
        for n in range(N):
            predictions.append(majority)


        return predictions
