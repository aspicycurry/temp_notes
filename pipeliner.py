import pandas as pd
import numpy as np
import cPickle as pickle
from pickler import build_bayest_pickle, nlp_transform
from imputer import impute_into_df
from munger import munge_dates
from htmler import html_features
from jsoner import add_features
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class DataPipeline(object):
    '''
    Class for transforming and cleaning data for the case study.
    '''

    def __init__(self, sample_size=None, event='test'):
        '''
        INPUT:
            sample_size -> int
            event -> str
        Initiates the class object.
        '''
        self.df = None
        self.nlp_cols = None
        self.sample_size = sample_size
        self.event = event

    def transform_data(self, df, nlp_cols=['description_content',
                                                  'org_desc_content', 'name',
                                                  'org_name', 'venue_name']):
        '''
        INPUT:
            json_file -> str of file path
            nlp_cols -> list of str of column names
        Reads in the assigned json file and calls data transformation methods.
        These methods will impute data, apply Naive Bayes to text data, clean
        and transform html and json data, and remove unneeded columns.
        '''
        self.df = df
        self.pickle_feature_headers()
        self.nlp_cols = nlp_cols
        self.update_features()
        self.drop_irrelevant_data()

    def pickle_feature_headers(self):
        if self.event == 'train':
            df = self.df.copy()
            df = df.drop('acct_type', axis=1)
            with open('data/features.pkl', 'w') as f:
                pickle.dump(list(df.columns), f)

    def update_features(self):
        '''
        Imputes data, apply Naive Bayes to text data, clean and transform html
        and json data.
        '''
        if self.sample_size is not None:
            self.df = self.df.sample(self.sample_size)

        self.create_label()
        self.pickle_countries()
        self.df = impute_into_df(self.df, self.get_countries())
        self.df = munge_dates(self.df)
        self.df = html_features(self.df, 'description', ['font-size', 'color',
                                                         'backgroud-color'])
        self.df = html_features(self.df, 'org_desc')
        for col in self.nlp_cols:
            self.nlp_helper(col)
        self.df = add_features(self.df)

    def pickle_countries(self):
        '''
        If the transformation is for training set, create a pickle file to keep
        track of the countries allowed for dummyfying.
        '''
        if self.event == 'train':
            country = set(list(self.df.country) + list(self.df.venue_country))
            country.remove(None)
            with open('data/countries.pkl', 'w') as f:
                pickle.dump(country, f)

    def create_label(self):
        '''
        If the transformation is for the training set, create a label column
        for frauds.
        '''
        if self.event == 'train':
            func = lambda x: 1 if 'fraud' in x else 0
            self.df['label'] = self.df.acct_type.apply(func)
            self.df = self.df.drop('acct_type', axis=1)

    def get_countries(self):
        with open('data/countries.pkl') as f_un:
            countries = pickle.load(f_un)
        return countries

    def nlp_helper(self, x_col):
        '''
        INPUT:
            x_col -> str of column name to perform Naive Bayes
            y_col -> str of column name to use as labels
        Uses Naive Bayes model to transform column to prediction values, create
        new pickle file of model if this is using training data.
        '''
        X = self.df[x_col]
        if self.event == 'train':
            y = self.df['label']
            build_bayest_pickle(X, y, 'data/{0}.pkl'.format(x_col))
        self.df = nlp_transform('data/{0}.pkl'.format(x_col), self.df, x_col)

    def drop_irrelevant_data(self):
        '''
        Drop unwanted columns (mostly due to time constraints).
        '''
        self.df = self.df.drop(['venue_latitude', 'venue_longitude', 'user_id',
                                'event_id', 'payee_name',
                                'venue_address', 'email_domain',
                                'venue_state', 'payout_type',
                                'listed'], axis=1)

    def pickle_data(self, file_path):
        '''
        INPUT:
            file_path -> str of path to create picke file.
        Takes the current dataframe and stores it into a pickle file.
        '''
        with open(file_path, 'w') as f:
            pickle.dump(self.df, f)
        print '{0} created!'.format(file_path)

    def get_data(self):
        '''
        Returns the current dataframe.
        '''
        return self.df

    def get_model(self):
        df = self.df.copy()
        clf = RandomForestClassifier(max_features='auto',
                                     n_estimators=200,
                                     class_weight='auto')
        y = df.pop('label')
        clf.fit(df, y)
        return clf

    def pickle_model(self, file_path='data/model.pkl'):
        model = self.get_model()
        with open(file_path, 'w') as f:
            pickle.dump(model, f)
        print '{0} created!'.format(file_path)
