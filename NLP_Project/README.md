Intent Classification and Out-Of-Scope Prediction


Here we have the code to test the accuracies and recall for four intent classification models: Convolutional Neural Network, BERT, MultiLayer Perceptron and Dialogflow.

Description of the dataset is given in the paper.pdf

CNN: Convolutional Neural Network is known for its edge case detectability and with poper assignement of hyperparameters we can achive the best accuracy. The above file CNN.ipynb can be used for all the datasets and parameters are to be tuned for each dataset.

Dialogflow: project_agent is a basic dialogue flow agent. data should be imported to the agent and then it can be used for intent recognition.

BERT.ipynb- has Bert Algorithm Implemented for Intent-classification of queries across small dataset. It can be used for all the datasets and parameters are to be tuned for each dataset. 

We used 'bert-large-uncased' pre-trained model for this task. you can download the BERT model using this link https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
 
MLP.py - has Multi-Layer Perceptron Algorithm Implemented for Intent-classification of queries across small dataset which gives in-scope accuracies and out-of-scope recall across the given dataset. 

Data_Preprocess.py, Model_Classification.py, Tokenization.py and Find_Accuracies.py- are implemented to enable integration of all algorithms. 
In this phase, we only integrated MLP algorithms with these files. All other Algorithm files run independently. 
However, in the next phase we would integrate all algorithms. 

Data_Preprocess.py- has implementation required for preprocessing of data. To allow reusability of pre-processing code for all algorithms
Model_Classification.py- allows user to pick the classification algorithm of their choice

Find_Accuracies.py- to find accuracies for the algorithms on all datasets
