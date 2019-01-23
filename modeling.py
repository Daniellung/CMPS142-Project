import time
import pandas as pd
from sklearn import linear_model, naive_bayes, neural_network, ensemble, multiclass
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced

def dual_part_model(train, test, models_neutrality, models_positivity):
    target = 'Neutrality'
    X_train = train[features]
    X_test = test[features]
    y_train = train[target]
    y_test = test[target]

    for model_name, model in models_neutrality.items():
        print(model_name + ' training starting now')
        time1 = time.time()
        model.fit(X_train, y_train)
        time2 = time.time()
        print('Model training took %f s'%(time2-time1))
        preds = model.predict(X_test)
        test['Pred ' + target] = preds
        print(classification_report(test['Pred ' + target], y_test))

    train = train[train['Sentiment'] != 2]
    target = 'Positivity'
    X_train = train[features]
    y_train = train[target]

    for model_name, model in models_positivity.items():
        print(model_name + ' training starting now')
        time1 = time.time()
        model.fit(X_train, y_train)
        time2 = time.time()
        print('Model training took %f s'%(time2-time1))
        preds = model.predict(X_test)
        test['Pred ' + target] = preds
        print(classification_report(test['Pred ' + target], y_test))

    test = test.drop(features, axis=1)

    test['Pred Sentiment'] = 2
    test['Pred Sentiment'][(test['Pred Positivity'] == 0) & (test['Pred Neutrality'] == 2)] = 0
    test['Pred Sentiment'][(test['Pred Positivity'] == 0) & (test['Pred Neutrality'] == 1)] = 1
    test['Pred Sentiment'][(test['Pred Positivity'] == 1) & (test['Pred Neutrality'] == 1)] = 3
    test['Pred Sentiment'][(test['Pred Positivity'] == 1) & (test['Pred Neutrality'] == 2)] = 4
    return test


def single_model(train, test, models):
    target = 'Sentiment'
    X_train = train[features]
    X_test = test[features]
    y_train = train[target]
    y_test = test[target]

    for model_name, model in models.items():
        print(model_name + ' training starting now')
        time1 = time.time()
        model.fit(X_train, y_train)
        time2 = time.time()
        print('Model training took %f s'%(time2-time1))
        test_probs = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
        test_probs[model_name + ' Pred Sentiment'] = test_probs.idxmax(axis=1)
        new_cols = [model_name + ' class ' + str(n) + ' prob ' for n in model.classes_]
        new_cols.append(model_name + ' Pred Sentiment')
        test_probs.columns = new_cols
        #test_probs.to_csv(model_name + 'predictions.csv', index=False)
        test = pd.concat([test.reset_index(drop=True), test_probs], axis=1, sort=False)
        print(classification_report(test[model_name + ' Pred Sentiment'], test['Sentiment']))
        print(accuracy_score(test[model_name + ' Pred Sentiment'], test['Sentiment']))

    test = test.drop(features, axis=1, errors='ignore')
    return test

df = pd.read_csv('train_cleaned.csv')

remove_cols = ['PhraseId', 'SentenceId', 'Phrase', 'Cleaned Phrase', 'Sentiment', 'Neutrality', 'Positivity']
features = [col for col in df.columns if col not in remove_cols]
s
train = df.sample(frac=0.9, random_state=42)
test = df.drop(train.index)

print('ready for modeling')
m1 = lgb.LGBMClassifier()
m2 = linear_model.LogisticRegression(C=.01, random_state=42)
m3 = linear_model.LogisticRegression(C=.001, random_state=42)

models = {
#'LogReg(chang.01)':multiclass.OneVsRestClassifier(linear_model.LogisticRegression(C=.01, random_state=42)),
#'LogReg(chang.001)':multiclass.OneVsRestClassifier(linear_model.LogisticRegression(C=.001, random_state=42)),
'NB':naive_bayes.GaussianNB(),
'GBM':lgb.LGBMClassifier(),
'LogReg(C=.01)':linear_model.LogisticRegression(C=.01, random_state=42),
'LogReg(C=.001)':linear_model.LogisticRegression(C=.001, random_state=42),
#'Ens':ensemble.VotingClassifier(estimators=[('1', m1),('2', m2),('3', m3)], voting='soft')
#'SGD':linear_model.SGDClassifier(loss='modified_huber', penalty='none')
'NN':neural_network.MLPClassifier(learning_rate = 'adaptive', hidden_layer_sizes = (1000,10), max_iter = 100, shuffle = True, verbose = False, warm_start = True, early_stopping = True),
}

models_neutrality = {
'GBM':lgb.LGBMClassifier(),
'LogReg(C=.01)':linear_model.LogisticRegression(C=.01, random_state=42),
#'LogReg(C=.001)':linear_model.LogisticRegression(C=.001, random_state=42),
}

models_positivity = {
'GBM':lgb.LGBMClassifier(),
'LogReg(C=.01)':linear_model.LogisticRegression(C=.01, random_state=42),
'LogReg(C=.001)':linear_model.LogisticRegression(C=.001, random_state=42),
}

submission = single_model(train, test, models)
#submission = dual_part_model(train, test, models_neutrality, models_positivity)
#print(classification_report(submission['Pred Sentiment'], submission['Sentiment']))
submission.to_csv('predictions.csv', index=False)
