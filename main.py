# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import plot_confusion_matrix

#%%
class selfSVM(object):
    def __init__(self, step_size = 0.1):
        self._estimator_type = "classifier"
        self.step_size = step_size
        return

    def extend_for_bias(self, inp):
        out = np.zeros((inp.shape[0],inp.shape[1]+1))
        out[:,:-1] = inp
        out[:,-1] = int(1)  
        return out

    def fit(self, data, labels, epochs = 10):
        data = self.extend_for_bias(data)
        self.weights = np.random.normal(loc=0, scale=0.05, size=data.shape[1])

        for _ in range(epochs): 
            for i in range(len(data)):
                Mar = labels[i]*np.dot(self.weights,data[i])

                if Mar >= 1:
                    self.weights = self.weights - 0.01 * self.step_size * self.weights / epochs
                else:
                    self.weights = self.weights + 0.01 * (labels[i] * data[i] - self.step_size * self.weights / epochs)

    def predict(self, data_in):
        outL = []
        data = self.extend_for_bias(data_in)
        for i in range(len(data)):
            outL.append(np.sign(np.dot(self.weights, data[i])))
        return np.array(outL)  

    def score(self,data, lables):
        prediction = self.predict(data)
        cnt = 0
        for i in range(len(lables)):
            if lables[i] == prediction[i]:
                cnt += 1
        return cnt / len(lables)

    def hinge_loss(self, x, y):
        return max(0,1 - y * np.dot(x, self.weights))

    def soft_margin_loss(self, x, y):
        return self.hinge_loss(x,y)+self.step_size * np.dot(self.weights, self.weights)

#%%
class MultiCalssificator(object):
    def __init__(self, classifier):
        self._estimator_type = "classifier"
        self.classifier = classifier
        return

    def devideDataBinary(self, data, labels, class_n):
        FClassData = []
        FClassLabels = []
        RestClassData = []
        RestClassLabels = []
        for i in range(len(data)):
            if labels[i] == class_n:
                FClassData.append(data[i])
                FClassLabels.append(class_n)
            else:
                RestClassData.append(data[i])
                RestClassLabels.append(labels[i])
        return FClassData, FClassLabels, RestClassData, RestClassLabels

    def fit(self, data, labels, classes_ord, epochs = 10):
        self.bin_classifiers = []
        self.classes_ord = classes_ord
        RestClassData, RestClassLabels = data, labels
        for i in range(len(classes_ord)):
            FClassData, FClassLabels, RestClassData, RestClassLabels = self.devideDataBinary(RestClassData, RestClassLabels, classes_ord[i])
            self.bin_classifiers.append(self.classifier())
            dataForBin = FClassData + RestClassData
            labelsForBin = [-1 for _ in range(len(FClassLabels))] + [1 for _ in range(len(RestClassLabels))]
            dataForBin, labelsForBin = shuffle(np.array(dataForBin), np.array(labelsForBin, dtype=int))
            self.bin_classifiers[i].fit(dataForBin, labelsForBin, epochs)
        return
    def predict_one(self, data_in):
        data = np.expand_dims(data_in, axis=0)
        i = 0
        for i in range(len(self.classes_ord)):
            lable = self.bin_classifiers[i].predict(data)[0]
            if lable == -1:
                lable = self.classes_ord[i]
                break
        return lable
    
    def predict(self, data_set):
        ans = []
        for data in data_set:
            ans.append(self.predict_one(data))
        return ans

    def score(self,data_set, lables):
        prediction = self.predict(data_set)
        cnt = 0
        for i in range(len(lables)):
            if lables[i] == prediction[i]:
                cnt += 1
        return cnt / len(lables)


#%%
class selfLogisticRegression(object):
    def __init__(self, learning_rate = 0.1):
        self._estimator_type = "classifier"
        self.learning_rate = learning_rate
        return

    def fit(self, data, labels, epochs = 100):
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = 0

        self.weights = np.random.normal(loc=0, scale=0.05, size=data.shape[1])
        for _ in range(epochs):  
            scores = np.dot(data, self.weights) 
            predictions = self.sigmoid(scores)  
            err = labels - predictions      
            gradient = np.dot(data.T, err)    
            self.weights += self.learning_rate * gradient 
        return

    def predict(self, data):
        predictions = self.predict_proba(data)
        for i in range(len(data)):
            if predictions[i] >= 0.5:
                predictions[i] = 1
            else:
                predictions[i] = -1
        return predictions 
    
    def predict_proba(self, data):
        scores = np.dot(data, self.weights) 
        predictions = self.sigmoid(scores) 
        return predictions 

    def score(self,data, lables):
        predictions = self.predict(data)
        cnt = 0
        for i in range(len(lables)):
            if lables[i] == predictions[i]:
                cnt += 1
        return cnt / len(lables)

    def sigmoid(self, scores): 
        return 1 / (1 + np.exp(-scores))

    def log_likelihood(data, labels, weights): 
        scores = np.dot(data, weights)  
        ll = np.sum(labels * scores - np.log(1 + np.exp(scores)))  
        return ll

#%%
class selfDecisionTreeClassifier(object):
    def __init__(self, max_depth, min_size):
        self._estimator_type = "classifier"
        self.root = None
        self.max_depth = max_depth
        self.min_size = min_size
        return

    def gini(self, data, labels, feature, value, classes):
        data_l, data_r, labels_l, labels_r = self.split(data, labels, feature, value)
        n = len(data)
        gini = 0
        for group, g_labels in [(data_l, labels_l), (data_r, labels_r)]:
            size = len(group)
            if size == 0:
                continue
            score = 0
            for class_n in classes:
                p = g_labels.count(class_n) / size
                score += p * p
            gini += (1.0 - score) * (size / n)
        return gini

    def split(self, data, labels, feature, value):
        data_l, data_r = [], []
        labels_l, labels_r = [], []
        for i in range(len(data)):
            if data[i][feature] < value:
                data_l.append(data[i])
                labels_l.append(labels[i])
            else:
                data_r.append(data[i])
                labels_r.append(labels[i])
        return data_l, data_r, labels_l, labels_r

    def leaf(self, lables):
        return {'leaf':True, 'class':max(set(lables), key=lables.count)}
        
    def decision_node(self, data, lables, classes):
        min_gini = 1000000000
        for i in range(len(data[0])):
            for entery in data:
                gini = self.gini(data, lables, i, entery[i], classes)
                if gini < min_gini:
                    index, value, min_gini = i, entery[i], gini
        return {'leaf':False, 'index':index, 'value':value, 'left':None, 'right':None}

    def build_sub_tree(self, data, labels, depth):
        classes = list(set(labels))
        if depth > self.max_depth or len(data) < self.min_size or len(classes)  == 1:
            return self.leaf(labels)
        depth += 1
        node = self.decision_node(data, labels, classes)
        data_l, data_r, labels_l, labels_r = self.split(data, labels, node['index'], node['value'])
        node['left'] = self.build_sub_tree(data_l, labels_l, depth)
        node['right'] = self.build_sub_tree(data_r, labels_r, depth)
        return node

    def fit(self, data, lables):
        self.root = self.build_sub_tree(data, lables, 1)
    
    def predict_one(self, fetures):
        node = self.root
        while True:
            if node['leaf']:
                return node['class']
            else:
                if fetures[node['index']] < node['value']:
                    node = node['left']
                else:
                    node = node['right']

    def predict(self, features_set):
        ans = []
        for features in features_set:
            ans.append(self.predict_one(features))
        return ans
    def score(self,features_set, lables):
        prediction = self.predict(features_set)
        cnt = 0
        for i in range(len(lables)):
            if lables[i] == prediction[i]:
                cnt += 1
        return cnt / len(lables)
    



# Dataset Loading
# %%
Star_train = np.load('Dataset/Star_train.npy')
Star_test = np.load('Dataset/Star_test.npy')
SLables_train = np.load('Dataset/SLables_train.npy')
SLables_test = np.load('Dataset/SLables_test.npy')



# SVM
# %%
classifier = MultiCalssificator(selfSVM)
classifier.fit(Star_train, SLables_train, [0, 1, 2, 3, 4, 5], 2000)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()

#%%
from sklearn import svm
classifier = svm.SVC()
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()



# SVM LogisticRegression
# %%
classifier = MultiCalssificator(selfLogisticRegression)
classifier.fit(Star_train, SLables_train, [0, 1, 2, 3, 4, 5], 30000)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()

# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=10000)
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()



# SVM DecisionTreeClassifier
# %%
classifier = selfDecisionTreeClassifier(10, 3)
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()

# %%
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(Star_train, SLables_train)
print(classifier.score(Star_test, SLables_test))
disp = plot_confusion_matrix(classifier, Star_test, SLables_test,
                                cmap=plt.cm.Blues,
                                normalize='true')
plt.show()
# %%
