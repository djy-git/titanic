from env import *
import preprocessing.feature_engineering as fe


if __name__ == "__main__":
    # Read train, test csv file
    train_data, test_data = pd.read_csv(TRAIN_CSV_PATH, encoding="utf8"), pd.read_csv(TEST_CSV_PATH, encoding="utf8")

    '''
    # Overview
    train_data.head()
    train_data.info()
    train_data.describe()
    train_data.hist(bins=50, figsize=(20, 15))
    '''

    # Manual processing (Remove rows whose value is Nan in ['Fare', 'Embarked', 'Age'] columns)
    train_data.dropna(subset=['Fare', 'Embarked', 'Age'], inplace=True)
    train_data.reset_index(drop=True, inplace=True)

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
    # Split with 'Age' value 11
    # X_train1, X_train2 = X_train[X_train['Age'] <= 11], X_train[X_train['Age'] > 11]
    # X_train1.reset_index(drop=True, inplace=True);  X_train2.reset_index(drop=True, inplace=True)

    train_data_proc, test_data_proc = fe.preprocess(train_data, train=True), fe.preprocess(test_data, train=False)
    # Columns
    # [Age  SibSp  Parch  log(Fare)  1st class  2nd class  3rd class  Female  Male  Miss  Mr  Mrs  B  C  D  E  Unknown]

    train_data_proc.to_csv(TRAIN_PROC_CSV_PATH, index=False)
    test_data_proc.to_csv(TEST_PROC_CSV_PATH, index=False)

    print("[Preprocessing] Preprocessed csv files are saved in")
    print(TRAIN_PROC_CSV_PATH)
    print(TEST_PROC_CSV_PATH)
