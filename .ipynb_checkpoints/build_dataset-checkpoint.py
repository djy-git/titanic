from env import *
from util import *

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


target = ['Survived']
cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
proc_num_cols = num_cols + ['log(Fare)', 'Household']
proc_Pclass = ['1st class', '2nd class', '3rd class']
cat_cols = ['Name', 'Sex', 'Ticket', 'Embarked']


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


if __name__ == "__main__":
    # Read train, test csv file
    TRAIN_CSV_PATH, TEST_CSV_PATH = "./data/original/train.csv", "./data/original/test.csv"
    train_data, test_data = pd.read_csv(TRAIN_CSV_PATH, encoding="utf8"), pd.read_csv(TEST_CSV_PATH, encoding="utf8")
    X_train, X_test = train_data.copy(), test_data.copy()
    y_train = X_train.drop(columns=target, inplace=True)

    # # Overview
    # train_data.head()
    # train_data.info()
    # train_data.describe()
    # train_data.hist(bins=50, figsize=(20, 15))

    # Remove ['Cabin', 'PassengerId'] columns
    X_train.drop(columns=['Cabin', 'PassengerId'], inplace=True)

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

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # DataFrame → ndarray
        ('num_feat_adder', NumFeatureAdder()),
        ('scaler', StandardScaler())
    ])

    # cat_pipeline = Pipeline([
    #
    # ])

    # Pipeline
    full_pipeline = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_cols),  # DataFrame → ndarray
        ('pclass_onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Pclass'])
        # ('cat_pipeline', cat_pipeline, cat_cols)
    ], n_jobs=-1)

    proc = full_pipeline.fit_transform(proc)
    proc = arr2df(proc, proc_num_cols + proc_Pclass)
    proc1, proc2 = full_pipeline.fit_transform(proc1), full_pipeline.fit_transform(proc2)
    proc1, proc2 = arr2df(proc1, proc_num_cols + proc_Pclass), arr2df(proc2, proc_num_cols + proc_Pclass)

