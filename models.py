from env import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, validation_curve
from scipy.stats import reciprocal, uniform


def print_cv_result(cvres):
    rst = pd.DataFrame(cvres)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_train_score', 'std_train_score']]
    sorted_rst = rst.sort_values(by='rank_test_score')
    print(sorted_rst)
    return


def random_cv(model, X, y, param, cv=5, n_iter=1000):
    rnd_search = RandomizedSearchCV(model, param, cv=cv, n_iter=n_iter, scoring='accuracy',
                                    return_train_score=True, n_jobs=-1, verbose=0)
    rnd_search.fit(X, y)
    print_cv_result(rnd_search.cv_results_)
    best_model = rnd_search.best_estimator_
    return best_model


def plot_validation_curve(model, X, y, param_name, param_range, scoring, cv=5):
    train_scores, val_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range,
                                                cv=cv, scoring=scoring, n_jobs=-1)
    train_scores_mean, train_scores_std = np.mean(train_scores, axis=-1), np.std(train_scores, axis=-1)
    val_scores_mean, val_scores_std = np.mean(val_scores, axis=-1), np.std(val_scores, axis=-1)

    plt.semilogx(param_range, train_scores_mean, label='Training score', color='r')
    plt.fill_between(param_range,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2, color='r')

    plt.semilogx(param_range, train_scores_mean, label='Validation score', color='g')
    plt.fill_between(param_range,
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std,
                     alpha=0.2, color='g')

    plt.title('Validation Curve')
    plt.xlabel(param_name);  plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend();  plt.grid()
    plt.show()

    opt_idx = np.argmax(val_scores_mean)
    opt_param, opt_score = param_range[opt_idx], val_scores_mean[opt_idx]
    print("Optimal param:", opt_param)
    print("Optimal score:", opt_score)

    return opt_param, opt_score


# Random forest
rnd_forest = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)

# SVM
svc = SVC()
param_SVC = {
    'gamma': reciprocal(1e-3, 1e-0),
    'C': uniform(1, 100)
}
