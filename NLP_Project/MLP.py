from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter


class MultiLayerPerceptron:

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):

        self.model = MLPClassifier(hidden_layer_sizes=(400, ), activation='tanh', solver='adam', alpha=0.0001, batch_size='auto',
                                   learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                   max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                                   warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                   validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.y_test_pred = None
        self.y_val_pred = None

        self.class_labels = list(set(self.y_train))

        self.assign_num_labels = {}
        for i in range(len(self.class_labels)):
            self.assign_num_labels[self.class_labels[i]] = i

        self.y_train = np.array([self.assign_num_labels[i] for i in self.y_train],dtype=float)
        self.y_Test = np.array([self.assign_num_labels[i] for i in self.y_Test],dtype=float)
        self.y_val = np.array([self.assign_num_labels[i] for i in self.y_val],dtype=float)

        # print(self.x_train[1:7])
        # print(type(self.y_train))
        # print(self.assign_num_labels)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def predict_test(self):
        self.y_test_pred = self.model.predict(self.x_test)
        self.y_val_pred = self.model.predict(self.x_val)
        print("predicted_vals",self.y_val_pred)


    def get_accuracy(self):
        label_num = self.assign_num_labels.get('oos')

        boolean_val = (self.y_val != label_num)
        y_val_inclass = self.y_val[self.y_val != label_num]
        y_val_pred_inclass = [self.y_val_pred[i] for i in range(len(self.y_val_pred)) if boolean_val[i]]

        boolean_test = (self.y_Test != label_num)
        y_test_inclass = self.y_Test[self.y_Test != label_num]
        y_test_pred_inclass = [self.y_test_pred[i] for i in range(len(self.y_test_pred)) if boolean_test[i]]

        return accuracy_score(y_true = y_test_inclass,y_pred = y_test_pred_inclass),accuracy_score(y_true = y_val_inclass,y_pred = y_val_pred_inclass)

    def get_out_of_scope_recall(self):

        label_num = self.assign_num_labels.get('oos')
        print(label_num)
        true_positives_test = Counter(np.logical_and(self.y_test_pred == label_num,self.y_Test == label_num))[True]
        false_negatives_test = Counter(np.logical_and(self.y_test_pred != label_num,self.y_Test == label_num))[True]


        true_positives_val = Counter(np.logical_and(self.y_val_pred == label_num,self.y_val == label_num))[True]
        false_negatives_val = Counter(np.logical_and(self.y_val_pred != label_num,self.y_val == label_num))[True]

        print(true_positives_val)
        print(false_negatives_val)
        val_recall = (true_positives_val)/(true_positives_val+false_negatives_val)
        test_recall = (true_positives_test)/(true_positives_test+false_negatives_test)

        return val_recall,test_recall



    def get_model(self):
        return self.model
