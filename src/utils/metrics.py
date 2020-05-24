import math
import scipy
from sklearn.metrics import r2_score, mean_squared_error


def pearson(y_t, y_p):
    """
    Determine the Pearson coefficient between the true values of an
    RPS parameter and the predicted values.

    Args:
        y_t (np.array):    Array (N, ) of true RPS values
        y_p (np.array):    Array (N, ) of predicted RPS values

    Returns:
        pearson_R:      Pearson correlation coefficient value
    """

    return scipy.stats.pearsonr(y_t, y_p)


def r2(y_t, y_p):
    return r2_score(y_t, y_p)


def rmse(y_t, y_p):
    return math.sqrt(mean_squared_error(y_t, y_p))
