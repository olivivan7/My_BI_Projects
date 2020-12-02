import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class AssumptionGraphs:

    def __init__(self, model):
        self.model = model
        self.model_norm_residuals = model.get_influence().resid_studentized_internal
        self.model_norm_residuals_abs_sqrt = np.sqrt(np.abs(self.model_norm_residuals))
        self.model_abs_resid = np.abs(model.resid)
        self.model_leverage = model.get_influence().hat_matrix_diag
        self.model_cooks = model.get_influence().cooks_distance[0]

    def plot_residual_fitted_values(self, y):
        sns.residplot(self.model.fittedvalues, y, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plt.title('Resíduos vs Ajustados', fontsize=20)
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Resíduos')

        top_3_abs_resid = self.model_abs_resid.sort_values(ascending=False)[:3]
        for index in top_3_abs_resid.index:
            plt.annotate(index, xy=(self.model.fittedvalues[index], self.model.resid[index]))

    def plot_scale_location(self):
        plt.scatter(self.model.fittedvalues, self.model_norm_residuals_abs_sqrt, alpha=0.5)
        sns.regplot(self.model.fittedvalues, self.model_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plt.title('Scale-Location', fontsize=20)
        plt.xlabel('Valores Ajustados')
        plt.ylabel('$\sqrt{|Resíduos\ Pradronizados|}$')

        top_3_abs_sq_norm_resid = np.flip(np.argsort(self.model_norm_residuals_abs_sqrt), axis=0)[:3]
        for i in top_3_abs_sq_norm_resid:
            plt.annotate(i, xy=(self.model.fittedvalues[i], self.model_norm_residuals_abs_sqrt[i]))

    def plot_influence(self):
        plt.scatter(self.model_leverage, self.model_norm_residuals, alpha=0.5)
        sns.regplot(self.model_leverage, self.model_norm_residuals,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        max_x = self.model_leverage.max() + 0.1 * self.model_leverage.max()
        plt.xlim(0, max_x)
        plt.ylim(-3, 4)
        plt.title('Residuals vs Leverage')
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residuals')
        plt.axhline(ls=':')

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.model_cooks), 0)[:3]

        for i in leverage_top_3:
            plt.annotate(i,
                         xy=(self.model_leverage[i],
                             self.model_norm_residuals[i]))

        # shenanigans for cook's distance contours
        def graph(formula, x_range, label=None):
            x = x_range
            y = formula(x)
            plt.plot(x, y, label=label, lw=1, ls='--', color='red')
            plt.plot(x, -y, label=label, lw=1, ls='--', color='red')

        p = len(self.model.params) # number of model parameters

        graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
            np.linspace(0.001, max_x, 50),
            'Cook\'s distance') # 0.5 line

        graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
            np.linspace(0.001, max_x, 50)) # 1 line

        plt.legend(loc='upper right')

    def plot_qq(self):
        stats.probplot(self.model.resid, dist="norm", plot=plt)