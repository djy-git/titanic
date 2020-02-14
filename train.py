from env import *
from Tuner import Tuner

if __name__ == "__main__":
    ########################
    ## 1. Load data
    ########################
    train_naive_proc_data, train_proc_data = pd.read_csv(TRAIN_NAIVE_PROC_CSV_PATH), pd.read_csv(TRAIN_PROC_CSV_PATH)
    test_naive_proc_data, test_proc_data = pd.read_csv(TEST_NAIVE_PROC_CSV_PATH), pd.read_csv(TEST_PROC_CSV_PATH)


    ########################
    ## 2. Preprocess data
    ########################
    X_train_naive, y_train_naive = split_target(train_naive_proc_data)
    X_train, y_train = split_target(train_proc_data)
    X_test_naive = test_naive_proc_data
    X_test = test_proc_data


    ########################
    ## 3. Model selection
    ########################
    ## 1) Model
    model = SVC()  # RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)

    ## 2) Tuning parameters with Cross-Validation
    tuner = Tuner(model)

    # # best_model = model.random_cv(model, X_train, y_train)
    tuner.plot_validation_curve(X_train, y_train,
                                param_name='gamma',
                                param_range=np.logspace(-4, 4, num=1000),
                                scoring='accuracy')

    tuner.plot_validation_curve(X_train, y_train,
                                param_name='C',
                                param_range=np.logspace(-1, 2, num=1000),
                                scoring='accuracy')

    # y_pred = model.predict(X_test)
    # X_test = pd.read_csv(TEST_CSV_PATH)['PassengerId']
    # y_pred = pd.Series(y_pred, name='Survived')
    # result = pd.concat([X_test, y_pred], axis=1)
    # print(result)
    # result.to_csv("submission.csv", index=False)