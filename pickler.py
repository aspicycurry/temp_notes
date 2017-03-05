from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
import cPickle as pickle

class BayestModel(object):
    
    def __init__(self):
        nb_grid = {'fit_prior': [True, False], 
                   'class_prior': [[.1, .9]],
                   'alpha': [1, 0.6, 0.9, 1.1, 1.3]
                  }
        self.nb_gridsearch = GridSearchCV(MultinomialNB(), nb_grid, 
                                          n_jobs=-1, scoring='roc_auc')

    def fit(self, X, y):
        self.vect = TfidfVectorizer(stop_words='english')
        self.mat = self.vect.fit_transform(X)
        self.nb_gridsearch.fit(self.mat, y)
        self.model = self.nb_gridsearch.best_estimator_

    def predict(self, x):
        vect_x = self.vect.transform([x])
        return self.model.predict(vect_x)


def build_bayest_pickle(X, y, file_path):
    model = BayestModel()
    model.fit(X, y)
    with open(file_path, 'w') as f:
        pickle.dump(model, f)
    print '{0} created!'.format(file_path)

def nlp_transform(pickle_path, df, col):
    predictions = []
    with open(pickle_path) as f_un:
        model = pickle.load(f_un)
    for row in df[col]:
        predictions.append(model.predict(row)[0])
    df[col] = predictions
    return df