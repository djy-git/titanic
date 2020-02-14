from env import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, validation_curve
from scipy.stats import reciprocal, uniform


class Tuner:
    def __init__(self, model):
        self.model = model
        self.params = self.model.get_params()
        self.init_cv_params = self._init_cv_params(type(self.model))

    def random_cv(self, X, y, params=None, cv=5, n_iter=1000, inplace=True):
        print("Parameters:", self.params)
        params = self.init_cv_params if params is None else params
        rnd_search = RandomizedSearchCV(self.model, params, cv=cv, n_iter=n_iter, scoring='accuracy',
                                        return_train_score=True, n_jobs=-1, verbose=0)
        rnd_search.fit(X, y)
        self._print_cv_result(rnd_search.cv_results_)
        best_model = rnd_search.best_estimator_
        if inplace:
            self.model = best_model
            return
        return best_model

    def plot_validation_curve(self, X, y, param_name, param_range, scoring, cv=5, inplace=True):
        train_scores, val_scores = validation_curve(self.model, X, y, param_name=param_name, param_range=param_range,
                                                    cv=cv, scoring=scoring, n_jobs=-1)
        train_scores_mean, train_scores_std = np.mean(train_scores, axis=-1), np.std(train_scores, axis=-1)
        val_scores_mean, val_scores_std = np.mean(val_scores, axis=-1), np.std(val_scores, axis=-1)

        # Plot with semi log
        plt.semilogx(param_range, train_scores_mean, label='Training score', color='r', lw=2)
        plt.semilogx(param_range, val_scores_mean, label='Validation score', color='g', lw=2)

        # Fill 95% confidence interval
        plt.fill_between(param_range,
                         train_scores_mean - 2*train_scores_std,
                         train_scores_mean + 2*train_scores_std,
                         alpha=0.2, color='r')
        plt.fill_between(param_range,
                         val_scores_mean - 2*val_scores_std,
                         val_scores_mean + 2*val_scores_std,
                         alpha=0.2, color='g')

        plt.title('Validation Curve')
        plt.xlabel(param_name);  plt.ylabel(scoring)
        plt.ylim(0, 1)
        plt.legend();  plt.grid();  plt.show()

        opt_idx = np.argmax(val_scores_mean)
        opt_param, opt_score = param_range[opt_idx], val_scores_mean[opt_idx]

        print("Optimal param:", opt_param)
        print("Optimal score:", opt_score)

        #TODO
        # Modification needed
        # if inplace:
        #     self.model.set_params(opt_param)
        #     return
        return opt_param, opt_score

    @staticmethod
    def _print_cv_result(cvres):
        rst = pd.DataFrame(cvres)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_train_score', 'std_train_score']]
        sorted_rst = rst.sort_values(by='rank_test_score')
        print(sorted_rst)
        return

    @staticmethod
    def _init_cv_params(type):
        if type == SVC:
            return {'C': np.logspace(-1, 2),
                    'gamma': np.logspace(-4, 4)}
        elif type == RandomForestClassifier:
            pass



