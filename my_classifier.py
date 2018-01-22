from sklearn.tree import DecisionTreeClassifier

class BatchClassifier:

#instancewindow()
    def __init__(self, window_size = 100, max_models = 10):
        self.H = []
        self.windowSize = window_size
        self.maxModels = max_models
        self.preData = []
        self.preY = []
        self.lastModelIndex = 0

    def assaignXtoPreData(self, X):
        self.preData = []
        self.addNdArrayElementsToPreList(X)

    def assaignYtoPreY(self, y):
        self.preY = []
        self.addNdArrayElementsToPreY(y)

    def addNdArrayElementsToPreList(self, ndArrray):
        for x in ndArrray:
            self.preData.append(x)

    def addNdArrayElementsToPreY(self, ndArrray):
        for x in ndArrray:
            self.preY.append(x)

    def addArrayElementsToList(self, array, ndArrray):
        for x in ndArrray:
            array.append(x)
        return array

    def addArrayElementsToY(self, array, ndArrray):
        for x in ndArrray:
            array.append(x)
        return array

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
        if X.size + len(self.preData) < self.windowSize :
            self.addNdArrayElementsToPreList(X)
            self.addNdArrayElementsToPreY(y)

        else:
            while (True):
                useFromX = self.windowSize - len(self.preData)
                useFromY = self.windowSize - len(self.preY)

                if len(self.preData) == 0:
                    xToUse = X[:useFromX]
                    yToUse = y[:useFromY]
                else:
                    xToUse = self.addArrayElementsToList(self.preData.copy(), X[:useFromX])
                    yToUse = self.addArrayElementsToY(self.preY.copy(), y[:useFromY])

                    self.preY = []
                    self.preData = []

                X = X[useFromX:]
                y = y[useFromY:]

                h = DecisionTreeClassifier()
                h.fit(xToUse, yToUse)
                self.addModel(h)

                if(X.size < self.windowSize):
                    self.assaignXtoPreData(X)
                    self.assaignYtoPreY(y)
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
        for n in range(D+1):
            predictions.append(majority)


        return predictions
