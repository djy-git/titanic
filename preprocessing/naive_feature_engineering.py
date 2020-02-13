import sys
sys.path.insert(1, '../')
from env import *

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Pclass', 'Sex']


class NumFeatureModifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_X = pd.DataFrame(X, columns=num_cols)
        df_X['Fare'] = np.log(df_X['Fare'] + 1)
        return df_X


class FeatureRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X_pip, y=None):
        onehot_cols = column_pipeline.named_transformers_['cat_pipeline'].get_feature_names(cat_cols).tolist()
        df_X_pip = pd.DataFrame(X_pip, columns=num_cols + onehot_cols)
        df_X_pip.rename(columns=self.columns, inplace=True)
        return df_X_pip


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('num_feat_modifier', NumFeatureModifier()),
    ('std_scaler', StandardScaler())
])

column_pipeline = ColumnTransformer([
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

naive_pipeline = Pipeline([
    ('column_pipline', column_pipeline),
    ('feat_renamer', FeatureRenamer(columns={'Fare': 'ln(Fare)'}))
])


def preprocess(data, train):
    y = data['Survived'] if train else pd.DataFrame()
    df_X_pip = naive_pipeline.fit_transform(data)
    return df_X_pip.join(y)
