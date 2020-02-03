from env import *
from util import *

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

np.set_printoptions(precision=2, edgeitems=20, linewidth=1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)

target = ['Survived']
cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

proc_num_cols = num_cols + ['log(Fare)', 'Household']
proc_Pclass = ['1st class', '2nd class', '3rd class']
proc_cat_cols = ['Sex', 'Title', 'Deck']        # ['Last name', 'Ticket', 'Embarked']
proc_sex_oh_cols = ['Female', 'Male']           # Caution Order!!
proc_title_oh_cols = ['Miss', 'Mr', 'Mrs']      # Caution Order!!
proc_deck_oh_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
proc_cat_oh_cols = proc_sex_oh_cols + proc_title_oh_cols + proc_deck_oh_cols
proc_cols = proc_num_cols + proc_Pclass + proc_cat_cols + target
proc_cols_oh = proc_num_cols + proc_Pclass + proc_cat_oh_cols + target


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
            # X.drop(columns='Fare', inplace=True)  # leave it
        if self.Household:
            X['Household'] = X['SibSp'] + X['Parch']
            # X.drop(columns=['SibSp', 'Parch'], inplace=True)  # leave it
        return X


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
        X.drop(columns=['Name', 'First name'], inplace=True)

        # Combine 'Title' into ['Mr', 'Mrs', 'Miss']
        X['Title'] = X.apply(self._combine_title, axis=1)

        # Extract 'Deck' from 'Cabin'
        X = self._extract_deck(X)
        X.drop(columns='Cabin', inplace=True)

        # Remove other variables
        X.drop(columns=['Last name', 'Ticket', 'Embarked'], inplace=True)

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
        return X

    @staticmethod
    def _find_substring(origin, ss_list):
        if type(origin) != str:
            return 'Unknown'
        for ss in ss_list:
            if str.find(origin, ss) != -1:
                return ss


if __name__ == "__main__":
    # Read train, test csv file
    TRAIN_CSV_PATH, TEST_CSV_PATH = "./data/original/train.csv", "./data/original/test.csv"
    train_data, test_data = pd.read_csv(TRAIN_CSV_PATH, encoding="utf8"), pd.read_csv(TEST_CSV_PATH, encoding="utf8")
    X_train, X_test = train_data.copy(), test_data.copy()
    # y_train = X_train.drop(columns=target, inplace=True)

    # # Overview
    # train_data.head()
    # train_data.info()
    # train_data.describe()
    # train_data.hist(bins=50, figsize=(20, 15))

    # Remove ['PassengerId'] columns
    X_train.drop(columns=['PassengerId'], inplace=True)

    # Remove rows whose value is Nan in ['Fare', 'Embarked', 'Age'] columns
    X_train.dropna(subset=['Fare', 'Embarked', 'Age'], inplace=True)
    X_train.reset_index(drop=True, inplace=True)

    '''
    > train_data
    RangeIndex: 712 entries, 0 to 711
    Data columns (total 10 columns):
    Survived    712 non-null int64
    Pclass      712 non-null int64
    Name        712 non-null object
    Sex         712 non-null object
    Age         712 non-null float64
    SibSp       712 non-null int64
    Parch       712 non-null int64
    Ticket      712 non-null object
    Fare        712 non-null float64
    Embarked    712 non-null object
    dtypes: float64(2), int64(4), object(4)
    '''

    ### Feature engineering
    proc = proc_X_train = X_train  # id(proc) = id(proc_X_train) = id(X_train)

    # Split with 'Age' value 11
    proc1, proc2 = proc[proc['Age'] <= 11], proc[proc['Age'] > 11]
    proc1.reset_index(drop=True, inplace=True);  proc2.reset_index(drop=True, inplace=True)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # DataFrame → ndarray
        ('num_feat_adder', NumFeatureAdder()),
        ('scaler', StandardScaler())
    ], )

    cat_pipeline = Pipeline([
        ('cat_feat_adder', CatFeatureAdder()),
        ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

    # Pipeline
    full_pipeline = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_cols),  # DataFrame → ndarray
        ('pclass_onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Pclass']),
        ('cat_pipeline', cat_pipeline, cat_cols)
    ], n_jobs=-1, remainder='passthrough')

    # # OneHotEncoding X
    # proc = full_pipeline.fit_transform(proc)
    # proc = arr2df(proc, proc_cols)
    # proc1, proc2 = full_pipeline.fit_transform(proc1), full_pipeline.fit_transform(proc2)
    # proc1, proc2 = arr2df(proc1, proc_cols), arr2df(proc2, proc_cols)

    # OneHotEncoding O
    proc = full_pipeline.fit_transform(proc)
    proc = arr2df(proc, proc_cols_oh)
    # proc1, proc2 = full_pipeline.fit_transform(proc1), full_pipeline.fit_transform(proc2)
    # proc1, proc2 = arr2df(proc1, proc_cols_oh), arr2df(proc2, proc_cols_oh)

    