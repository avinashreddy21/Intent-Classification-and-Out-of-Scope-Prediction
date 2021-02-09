from Model_Classification import ModelClassification
from Data_Preprocess import get_split_data,get_in_scope_data
import numpy as np



''' Getting the data which is extracted using the Function defined in DataPreprocess.py '''

def predictions(dataset,model):

    train_df,val_df,test_df,inscope_train,inscope_val,inscope_test = get_split_data(dataset)


    if model == 'MLP':

        """ Training the data on 'MLP' Classifier """

        MLP_inscope = ModelClassification(model, train_df, test_df, val_df)
        val_accuracy, test_accuracy, val_recall, test_recall = MLP_inscope.get_metrics_for_model()
        print("Validation in-scope Accuracies for {} across {} is {}".format(model,dataset,val_accuracy))
        print("Test in-scope Accuracies for {} across {} is {}".format(model,dataset,test_accuracy))


        print("Validation Out_of_scope Recall for {} across {} is {}".format(model,dataset,val_recall))
        print("Test Out_of_scope Recall for {} across {} is {}".format(model,dataset,test_recall))

    elif model == 'MLP_keras':

        """ Training the data on 'MLP_keras' Classifier """

        MLP_inscope = ModelClassification(model, train_df, test_df, val_df)
        val_accuracy, test_accuracy, val_recall, test_recall = MLP_inscope.get_metrics_for_model()
        print("Validation in-scope Accuracies for {} across {} is {}".format(model,dataset,val_accuracy))
        print("Test in-scope Accuracies for {} across {} is {}".format(model,dataset,test_accuracy))


        print("Validation Out_of_scope Recall for {} across {} is {}".format(model,dataset,val_recall))
        print("Test Out_of_scope Recall for {} across {} is {}".format(model,dataset,test_recall))


    elif model == 'SVM':

        """ Training the data on 'SVM' Classifier """

        MLP_inscope = ModelClassification(model, train_df, test_df, val_df)
        val_accuracy, test_accuracy, val_recall, test_recall = MLP_inscope.get_metrics_for_model()
        print("Validation in-scope Accuracies for {} across {} is {}".format(model,dataset,val_accuracy))
        print("Test in-scope Accuracies for {} across {} is {}".format(model,dataset,test_accuracy))


        print("Validation Out_of_scope Recall for {} across {} is {}".format(model,dataset,val_recall))
        print("Test Out_of_scope Recall for {} across {} is {}".format(model,dataset,test_recall))



if __name__ == '__main__':

    datasets = ['full','small','imbal','oos+']
    models = ['MLP_keras']

    for data in datasets:
        for model in models:
            predictions(data,model)




