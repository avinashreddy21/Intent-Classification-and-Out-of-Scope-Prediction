from MLP import MultiLayerPerceptron
from SVM import SupportVectorMachine
from MLP_keras import MultiLayerPerceptron_Keras
import gensim
from gensim.models import Doc2Vec, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
import nltk
from Tokenization import extract_features
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from nltk.corpus import stopwords
import multiprocessing
from bs4 import BeautifulSoup
import re

# nltk.download('all')


class ModelClassification:


    def __init__(self, model_name, train_df, test_df, val_df):

        # def tokenize_text(text):
        #     tokens = []
        #     for sent in nltk.sent_tokenize(text):
        #         for word in nltk.word_tokenize(sent):
        #             if len(word) < 2:
        #                 continue
        #             tokens.append(word.lower())
        #     return tokens
        #
        # train_tagged = train_df.apply(
        #     lambda r: TaggedDocument(words=tokenize_text(r['query']), tags=[r['intent']]), axis=1)
        #
        # print(train_tagged.values[30])
        #
        # val_tagged = val_df.apply(
        #     lambda r: TaggedDocument(words=tokenize_text(r['query']), tags=[r['intent']]), axis=1)
        #
        # test_tagged = test_df.apply(
        #     lambda r: TaggedDocument(words=tokenize_text(r['query']), tags=[r['intent']]), axis=1)
        #
        # cores = multiprocessing.cpu_count()
        #
        # model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
        #                      hs=0, min_count=2, sample = 0, workers=cores)
        #
        # # model_dbow = Word2Vec(size=300, min_count=10, workers=cores,negative=5)
        #
        # model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
        #
        # for epoch in range(30):
        #     model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=2)
        #     model_dbow.alpha -= 0.002
        #     model_dbow.min_alpha = model_dbow.alpha
        #
        # def vec_for_learning(model, tagged_docs):
        #     sents = tagged_docs.values
        #     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        #     return targets, regressors
        #
        #
        # self.y_train, self.x_train = vec_for_learning(model_dbow, train_tagged)
        # self.y_val, self.x_val = vec_for_learning(model_dbow, val_tagged)
        # self.y_test, self.x_test = vec_for_learning(model_dbow, test_tagged)


        # GET LABELS
        self.y_train = train_df['intent'].values
        self.y_val = val_df['intent'].values
        self.y_test = test_df['intent'].values

        # GET FEATURES
        self.x_train,self.x_test,self.x_val = extract_features(field='query',training_data=train_df,
                                                                                       validation_data=val_df,
                                                                                       testing_data=test_df, type='tf')

        # self.train = train_df.values
        # self.validation = val_df.values
        # self.test = test_df.values

        # self.x_train, self.y_train = self.train[:, :-1], self.train[:,-1]
        # self.x_val, self.y_val = self.validation[:,:-1], self.validation[:,-1]
        # self.x_test, self.y_test = self.test[:,:-1], self.test[:,-1]

        self.model_name = model_name
        self.model = None
        self.accuracy_test, self.accuracy_val = None, None

        print('train-data-shape',self.x_train.shape)

    """ Model to call corresponding models """
    def get_metrics_for_model(self):

        if self.model_name == 'MLP':
            self.model = MultiLayerPerceptron(self.x_train, self.y_train,
                                              self.x_test, self.y_test,
                                              self.x_val, self.y_val)
            self.model.train_model()
            self.model.predict_test()
            self.accuracy_test, self.accuracy_val = self.model.get_accuracy()
            self.recall_val, self.recall_test = self.model.get_out_of_scope_recall()

        elif self.model_name == 'MLP_keras':

            self.model = MultiLayerPerceptron_Keras(self.x_train, self.y_train,
                                              self.x_test, self.y_test,
                                              self.x_val, self.y_val)
            self.model.train_model()
            self.model.predict_test()
            self.accuracy_test, self.accuracy_val = self.model.get_accuracy()
            self.recall_val, self.recall_test = self.model.get_out_of_scope_recall()

        elif self.model_name == 'BERT':
            pass

        elif self.model_name == 'SVM':
            self.model = SupportVectorMachine(self.x_train, self.y_train,
                                              self.x_test, self.y_test,
                                              self.x_val, self.y_val)
            self.model.train_model()
            self.model.predict_test()
            self.accuracy_test, self.accuracy_val = self.model.get_accuracy()
            self.recall_val, self.recall_test = self.model.get_out_of_scope_recall()


        elif self.model_name == 'CNN':
            pass

        elif self.model_name == 'DialogFlow':
            pass

        return self.accuracy_val,self.accuracy_test, self.recall_val, self.recall_test



