from models.env import *

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


target = ['Survived']
cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
num_cols = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

proc_num_cols = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare'] + ['log(Fare)']
proc_num_final_cols = ['Age', 'SibSp', 'Parch', 'log(Fare)']
proc_pclass_oh_cols = ['1st class', '2nd class', '3rd class']

name_oh_cols = ['First name', 'Last name', 'Title']
title_oh_cols = ['Miss', 'Mr', 'Mrs']
sex_oh_cols = ['Female', 'Male']
deck_oh_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

proc_cat_cols = cat_cols + name_oh_cols + ['Deck']
proc_cat_oh_cols = sex_oh_cols + title_oh_cols + deck_oh_cols
proc_cat_final_oh_cols = sex_oh_cols + title_oh_cols + ['B', 'C', 'D', 'E', 'Unknown']
proc_cols_oh = proc_num_final_cols + proc_pclass_oh_cols + proc_cat_final_oh_cols


class NumFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, log_Fare=True, Household=True):
        self.log_Fare, self.Household = log_Fare, Household
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=num_cols)  # ColumnTransformer() makes DataFrame to ndarray

        # Add 'log(Fare)' and 'Household'
        if self.log_Fare:
            X['log(Fare)'] = np.log(1 + X['Fare'])
        if self.Household:
            # X['Household'] = X['SibSp'] + X['Parch']
            # X.drop(columns=['SibSp', 'Parch'], inplace=True)  # leave it
            pass
        return X


class NumFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_X = pd.DataFrame(X, columns=proc_num_cols)

        # Remove 'out_cols' columns
        out_cols = ['PassengerId', 'Fare']
        df_X.drop(columns=out_cols, inplace=True)
        return df_X


class CatFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=cat_cols)

        # Split 'Name' into ['Last name', 'Title', 'First name']
        names = self._split_name(X)
        X = X.join(names)

        # Combine 'Title' into ['Mr', 'Mrs', 'Miss']
        X['Title'] = X.apply(self._combine_title, axis=1)

        # Extract 'Deck' from 'Cabin'
        X = self._extract_deck(X)
        return X

    def _split_name(self, X):
        names = pd.DataFrame(columns=['Last name', 'Title', 'First name'])
        for name in X['Name']:
            elems = re.split("[,.]", name)  # ['Last name', 'Title', 'First name']
            elems = [elem.strip() for elem in elems]
            elems = elems[:3]  # cut off in 3 names
            names.loc[len(names)] = elems
        return names

    @staticmethod
    def _combine_title(X):
        title = X['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Master']:
            return 'Mr'
        elif title in ['Countess', 'Mme', 'the Countess']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Lady']:
            return 'Miss'
        elif title == 'Dr':
            if X['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    def _extract_deck(self, X):
        cabin_list = np.unique([cabin[0] for cabin in X['Cabin'].unique() if type(cabin) != float])
        cabin_list = np.append(cabin_list, 'Unknown')
        X['Deck'] = X['Cabin'].map(lambda cabin: self._find_substring(cabin, cabin_list))
        X['Cabin'] = X['Cabin'].fillna(value='Unknown')
        return X

    @staticmethod
    def _find_substring(origin, ss_list):
        if type(origin) != str:
            return 'Unknown'
        for ss in ss_list:
            if str.find(origin, ss) != -1:
                return ss


class CatFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, flag):
        self.flag = flag
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_X, out_cols = None, None
        if self.flag == 1:
            df_X = X
            out_cols = ['Name', 'Ticket', 'Cabin', 'Embarked', 'First name', 'Last name']
        elif self.flag == 2:
            df_X = pd.DataFrame(X, columns=proc_cat_oh_cols)
            out_cols = ['A', 'F', 'T', 'G']
        df_X.drop(columns=out_cols, inplace=True)
        if self.flag == 2:
            global proc_cat_final_oh_cols
            proc_cat_final_oh_cols = df_X.columns
        return df_X


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # DataFrame → ndarray
    ('num_feat_adder', NumFeatureAdder()),
    ('num_feat_dropper', NumFeatureDropper()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('cat_feat_adder', CatFeatureAdder()),
    ('cat_feat_dropper1', CatFeatureDropper(flag=1)),
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore')),
    ('cat_feat_dropper2', CatFeatureDropper(flag=2)),
])

# Pipeline
full_pipeline = ColumnTransformer([
    ('num_pipeline', num_pipeline, num_cols),  # DataFrame → ndarray
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Pclass']),
    ('cat_pipeline', cat_pipeline, cat_cols)
], n_jobs=-1)


def preprocess(data, train=True):
    if train:
        y = data['Survived']
        X = data.drop(columns='Survived')
        X = full_pipeline.fit_transform(X)
        X = pd.DataFrame(X, columns=proc_cols_oh)
        return X.join(y)
    else:
        X = full_pipeline.transform(data)
        X = pd.DataFrame(X, columns=proc_cols_oh)
        return X
