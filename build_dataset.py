from env import *
import preprocessing.feature_engineering as fe
import preprocessing.naive_feature_engineering as nfe


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

    ### Feature engineering
    # Naive preprocess & Full process
    for preprocess, train_csv_path, test_csv_path in zip([nfe.preprocess, fe.preprocess],
                                                         [TRAIN_NAIVE_PROC_CSV_PATH, TRAIN_PROC_CSV_PATH],
                                                         [TEST_NAIVE_PROC_CSV_PATH, TEST_PROC_CSV_PATH]):
        train_data_proc = preprocess(train_data, train=True)
        test_data_proc = preprocess(test_data, train=False)

        #TODO
        # Split with 'Age' value 11
        # X_train1, X_train2 = X_train[X_train['Age'] <= 11], X_train[X_train['Age'] > 11]
        # X_train1.reset_index(drop=True, inplace=True);  X_train2.reset_index(drop=True, inplace=True)
        train_data_proc.to_csv(train_csv_path, index=False)
        test_data_proc.to_csv(test_csv_path, index=False)

        print("[Preprocessing] csv files are saved in")
        print(train_csv_path)
        print(test_csv_path, end='\n\n')
