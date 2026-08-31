"""
Stricter, complete linear separation checks for features in a dataset against
    categorical/binary response variable.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog


@dataclass
class CompleteSeparationResult:
    separable: bool
    status: int
    message: str
    coefficients: pd.Series | None = None


def find_single_feature_separation_breakers(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> list[str]:
    """
    Identify features in a completely separable dataset whose removal breaks complete separation.
    NO-op (returns an empty list) if the full feature matrix is not completely separable.

    :param X: Feature matrix as a pandas DataFrame.
    :param y: Response variable as a pandas Series or numpy array.
    :return: List of feature names whose removal breaks complete separation.
    """

    baseline = _check_complete_separation(X, y)

    # Nothing needs correcting if the full model is already nonseparable.
    if not baseline.separable:
        return []

    breakers = []

    for feature in X.columns:
        reduced_result = _check_complete_separation(
            X.drop(columns=[feature]),
            y,
        )

        # This feature is a true breaker only because:
        #   full X      = separable
        #   X minus f   = nonseparable
        if not reduced_result.separable:
            breakers.append(feature)

    return breakers


def _check_complete_separation(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> CompleteSeparationResult:
    """
    Test for multivariate complete linear separation.

    :param X: Feature matrix as a pandas DataFrame.
    :param y: Response variable as a pandas Series or numpy array.
    :return: CompleteSeparationResult indicating if complete separation exists.

    Finds beta satisfying:

        y_i* (x_i @ beta) >= 1

    for every observation, where y_i* is {-1, +1}.
    """
    X = X.apply(pd.to_numeric, errors="raise")
    X = sm.add_constant(X, has_constant="add")

    y_arr = np.asarray(y)
    if set(np.unique(y_arr)) - {0, 1}:
        raise ValueError("y must be coded as 0/1.")

    y_pm = np.where(y_arr == 1, 1.0, -1.0)
    X_arr = X.to_numpy(dtype=float)

    result = linprog(
        c=np.zeros(X_arr.shape[1]),
        A_ub=-(y_pm[:, None] * X_arr),
        b_ub=-np.ones(len(y_pm)),
        bounds=[(None, None)] * X_arr.shape[1],
        method="highs",
    )

    coefs = None
    if result.success:
        coefs = pd.Series(result.x, index=X.columns)

    return CompleteSeparationResult(
        separable=result.success,
        status=result.status,
        message=result.message,
        coefficients=coefs,
    )
