from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Data_Preprocess import get_split_data
from Tokenization import extract_features
import tensorflow.contrib.keras as keras
import tensorflow as tf



class MultiLayerPerceptron_Keras:

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):

        self.class_labels = list(set(y_train))

        np.random.seed(123)
        tf.set_random_seed(123)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train)
        y_train = self.label_encoder.transform(y_train)


        # initialize model
        self.model = keras.models.Sequential()

        # add input layer
        self.model.add(keras.layers.Dense(
            units=50,
            input_dim=x_train.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh')
        )

        # add hidden layer
        self.model.add(
            keras.layers.Dense(
                units=50,
                input_dim=50,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='tanh')
            )
        print("y_train_shape",y_train.shape)

        # add output layer
        self.model.add(
            keras.layers.Dense(
                units=151,
                input_dim=50,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax')
            )

        # define SGD optimizer
        sgd_optimizer = keras.optimizers.SGD(
            lr=0.001, decay=1e-7, momentum=0.9
        )
        # compile model
        self.model.compile(
            optimizer=sgd_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.y_test_pred = None
        self.y_val_pred = None


        # print(self.x_train[1:7])
        # print(type(self.y_train))
        # print(self.assign_num_labels)

    def train_model(self):
        self.model.fit(
        self.x_train, self.y_train,
        batch_size=32, epochs=50,
        verbose=1, validation_split=0.1)

    def predict_test(self):
        self.y_test_pred = self.model.predict(self.x_test,verbose=0)
        self.y_val_pred = self.model.predict(self.x_val,verbose=0)
        print("predicted_vals",self.y_val_pred)


    def get_accuracy(self):
        # self.assign_num_labels = {}
        #
        # for i in range(len(self.class_labels)):
        #     self.assign_num_labels[self.class_labels[i]] = i
        #
        # self.y_train = np.array([self.assign_num_labels[i] for i in self.y_train],dtype=float)
        # self.y_Test = np.array([self.assign_num_labels[i] for i in self.y_Test],dtype=float)
        # self.y_val = np.array([self.assign_num_labels[i] for i in self.y_val],dtype=float)

        # label_num = self.assign_num_labels.get('oos')

        self.y_val_pred = self.label_encoder.inverse_transform(self.y_val_pred)
        self.y_test_pred = self.label_encoder.inverse_transform(self.y_test_pred)

        boolean_val = (self.y_val != 'oos')
        y_val_inclass = self.y_val[self.y_val != 'oos']
        y_val_pred_inclass = [self.y_val_pred[i] for i in range(len(self.y_val_pred)) if boolean_val[i]]

        boolean_test = (self.y_Test != 'oos')
        y_test_inclass = self.y_Test[self.y_Test != 'oos']
        y_test_pred_inclass = [self.y_test_pred[i] for i in range(len(self.y_test_pred)) if boolean_test[i]]

        return accuracy_score(y_true = y_test_inclass,y_pred = y_test_pred_inclass),accuracy_score(y_true = y_val_inclass,y_pred = y_val_pred_inclass)

    def get_out_of_scope_recall(self):

        label_num = 'oos'
        true_positives_test = Counter(np.logical_and(self.y_test_pred == label_num,self.y_Test == label_num))[True]
        false_negatives_test = Counter(np.logical_and(self.y_test_pred != label_num,self.y_Test == label_num))[True]


        true_positives_val = Counter(np.logical_and(self.y_val_pred == label_num,self.y_val == label_num))[True]
        false_negatives_val = Counter(np.logical_and(self.y_val_pred != label_num,self.y_val == label_num))[True]

        val_recall = (true_positives_val)/(true_positives_val+false_negatives_val)
        test_recall = (true_positives_test)/(true_positives_test+false_negatives_test)

        return val_recall,test_recall



    def get_model(self):
        return self.model
