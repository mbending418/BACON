import math
import numpy as np
from dataclasses import dataclass
from functools import cached_property


@dataclass(init=True, frozen=True)
class LinearModelPrediction:
    """
    Linear Regression model between y_data and x_data
    based of off regression.linear_regression.LinearModel from:
        https://github.com/mbending418/Regression

    x_data should be a 2D np.typing.NDArray
    every row of x_data should be a different predictor
    every row of x_data should be a data point

    x_data can be 1D if you only have one predictor

    y_data should be a 1D np.typing.NDArray
    with the same length as there are rows in x_data
    """

    x_data: np.typing.NDArray
    y_data: np.typing.NDArray

    def __post_init__(self):
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise Exception(
                f"x_data and y_data need to have the same number of data points:"
                f" X.shape={self.x_data.shape} | Y.shape={self.y_data.shape}"
            )

        if len(self.x_data.shape) > 2:
            raise Exception(
                f"x_data needs to be one or two dimensional: x_data.shape={self.x_data.shape}"
            )

        if len(self.y_data.shape) != 1:
            raise Exception(
                f"y_data needs to be one dimensional: y_data.shape={self.y_data.shape}"
            )

    @cached_property
    def n(self) -> int:
        """
        :return: number of data points
        """
        return self.x_data.shape[0]

    @cached_property
    def predictor_count(self) -> int:
        """
        number of predictor variables
        commonly denoted as "p"
        :return: number of predictors
        """
        if len(self.x_data.shape) == 1:
            return 1
        return self.x_data.shape[1]

    @cached_property
    def parameter_count(self) -> int:
        """
        number of model parameters/coefficients
        commonly demoted as "k"
        :return: number of model parameters/coefficients
        """
        return self.predictor_count + 1

    @cached_property
    def df(self) -> int:
        """
        degrees of freedom
        df = n - p - 1
        :return: degrees of freedom
        """
        return self.n - self.parameter_count

    @cached_property
    def design_matrix(self) -> np.typing.NDArray:
        """
        Design Matrix
        commonly denoted X
        X := [1,x]
        where 1 is a column vector of 1's
        where x is the column vector of x_data
        :return: Design Matrix
        """
        x_data = self.x_data
        if len(x_data.shape) == 1:
            x_data = x_data[..., np.newaxis]
        return np.concat((np.ones((x_data.shape[0], 1)), x_data), axis=1)

    @cached_property
    def c_matrix(self) -> np.typing.NDArray:
        """
        C Matrix
        This is an intermediate matrix used in several calculations
        C_matrix = inverse of design_matrix.transpose()*design_matrix
        :return: Inverse(Design_Matrix.transpose() * Design_Matrix
        """
        return np.linalg.inv(self.design_matrix.transpose() @ self.design_matrix)

    @cached_property
    def projection_matrix(self) -> np.typing.NDArray:
        """
        Projection Matrix also known as Hat Matrix
        Commonly denoted P or H
        this is the matrix such that y_hat = P*y
        P_matrix := X_matrix * C_matrix * X_matrix.transpose()
        :return: Projection Matrix
        """
        return self.design_matrix @ self.c_matrix @ self.design_matrix.transpose()

    @cached_property
    def beta_hat(self) -> np.typing.NDArray:
        """
        estimated model coefficients
        beta_hat := C_matrix * X_matrix.transpose * y_data
        :return: beta_hat
        """
        return self.c_matrix @ self.design_matrix.transpose() @ self.y_data

    @cached_property
    def y_hat(self) -> np.typing.NDArray:
        """
        fitted values for response variable
        y_hat := P*y
        :return: y_hat
        """
        return self.projection_matrix @ self.y_data

    @cached_property
    def residuals(self) -> np.typing.NDArray:
        """
        the residuals of our response variable
        residuals := y_data - y_fitted = y_data - P*y_data
        :return: residuals
        """
        return self.y_data - self.y_hat

    @cached_property
    def sse(self) -> float:
        """
        Sum of Squared Errors
        SSE := sum(residuals**2)
        :return: SSE
        """
        return float(sum(self.residuals**2))

    @cached_property
    def sigma_hat_squared(self) -> float:
        """
        estimate for model noise
        sigma_hat_squared := mse
        :return: sigma_hat_squared
        """
        return self.sse / self.df

    def predict(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the predicted value of the model at x0

        x0 : input x-value
        x0 should be an NDArray of predictors the size of predictor_count
        if you only have one predictor x0 can be a float
        :return: fitted model value at x0
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        if x0.shape != (self.predictor_count,):
            raise Exception(
                f"wrong shape for x0. Expected=({self.predictor_count},) Actual={x0.shape}"
            )
        x0 = np.concat((np.array([1]), x0))
        return float((x0 @ self.beta_hat))

    def predicted_value_standard_error(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the standard error for the predicted value of the model at x0

        :param x0: input x-value
        x0 should be an NDArray of predictors the size of predictor_count
        if you only have one predictor x0 can be a float
        :return: standard error for predicted value
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        if x0.shape != (self.predictor_count,):
            raise Exception(
                f"wrong shape for x0. Expected=({self.predictor_count},) Actual={x0.shape}"
            )
        x0 = np.concat((np.array([1]), x0))
        return float(math.sqrt(self.sigma_hat_squared * (1 + x0 @ self.c_matrix @ x0)))

    def residual_standard_error(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the standard error for the residual of the model at x0

        :param x0: input x-value
        :return: standard error for residual
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        if x0.shape != (self.predictor_count,):
            raise Exception(
                f"wrong shape for x0. Expected=({self.predictor_count},) Actual={x0.shape}"
            )
        x0 = np.concat((np.array([1]), x0))
        return float(math.sqrt(self.sigma_hat_squared * (1 - x0 @ self.c_matrix @ x0)))
