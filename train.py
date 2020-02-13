from env import *
import models

if __name__ == "__main__":
    ### Load data
    train_naive_proc_data = pd.read_csv(TRAIN_NAIVE_PROC_CSV_PATH)
    train_proc_data = pd.read_csv(TRAIN_PROC_CSV_PATH)

    test_naive_proc_data = pd.read_csv(TEST_NAIVE_PROC_CSV_PATH)
    test_proc_data = pd.read_csv(TEST_PROC_CSV_PATH)

    # ### Use naive proc data
    # X_train, y_train = split_target(train_naive_proc_data)
    # X_test = test_naive_proc_data

    ### Use full proc data
    X_train, y_train = split_target(train_proc_data)
    X_test = test_proc_data

    ### Model selection
    model = models.svc
    best_model = models.random_cv(model, X_train, y_train, param=models.param_SVC)
    # models.plot_validation_curve(model, X_train, y_train,
    #                              param_name='gamma',
    #                              param_range=np.logspace(-4, 4, num=1000),
    #                              scoring='accuracy')

    # models.plot_validation_curve(model, X_train, y_train,
    #                                  param_name='C',
    #                                  param_range=np.linspace(0.1, 100, num=1000),
    #                                  scoring='accuracy')

    # model = models.SVC(gamma=0.823978568452852)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    y_pred = best_model.predict(X_test)
    X_test = pd.read_csv(TEST_CSV_PATH)['PassengerId']
    y_pred = pd.Series(y_pred, name='Survived')
    result = pd.concat([X_test, y_pred], axis=1)
    print(result)
    result.to_csv("submission.csv", index=False)