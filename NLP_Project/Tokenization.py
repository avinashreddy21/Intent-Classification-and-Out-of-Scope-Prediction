from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def extract_features(field, training_data, validation_data, testing_data, type="binary"):

    print("Extracting features")

    if type == 'binary':

        # BINARY FEATURE REPRESENTATION
        cv_model= CountVectorizer(binary=True, max_df=0.95)
        cv_model.fit_transform(training_data[field].values)

        train_feature_set=cv_model.transform(training_data[field].values)
        test_feature_set=cv_model.transform(testing_data[field].values)
        validation_feature_set=cv_model.transform(validation_data[field].values)

        return train_feature_set,test_feature_set,validation_feature_set

    elif type == 'not_binary':

        # COUNT BASED FEATURE REPRESENTATION
        cv_model= CountVectorizer(binary=False, max_df=0.95)
        cv_model.fit_transform(training_data[field].values)

        train_feature_set=cv_model.transform(training_data[field].values)
        test_feature_set=cv_model.transform(testing_data[field].values)
        validation_feature_set=cv_model.transform(validation_data[field].values)

        return train_feature_set,test_feature_set,validation_feature_set

    else:

        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_model=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_model.fit_transform(training_data[field].values)

        train_feature_set=tfidf_model.transform(training_data[field].values)
        test_feature_set=tfidf_model.transform(testing_data[field].values)
        validation_feature_set=tfidf_model.transform(validation_data[field].values)

        return train_feature_set,test_feature_set,validation_feature_set
