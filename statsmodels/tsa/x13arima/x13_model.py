"""
Run x12/x13-arima specs in a subprocess from Python and curry results back
into python.
An alternative x13-arima implementation which no longer relies on external binaries.

Notes
-----
"""

from statsmodels.compat.pandas import deprecate_kwarg

import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import X13Result


class X13Model:
    def __init__(
        self,
        endog,
        model="additive",
        filt=None,
        period=None,
        two_sided=True,
        extrapolate_trend=0,
    ):
        """
        Seasonal decomposition using moving averages.

        Parameters
        ----------
        x : array_like
            Time series. If 2d, individual series are in columns. x must contain 2
            complete cycles.
        model : {"additive", "multiplicative"}, optional
            Type of seasonal component. Abbreviations are accepted.
        filt : array_like, optional
            The filter coefficients for filtering out the seasonal component.
            The concrete moving average method used in filtering is determined by
            two_sided.
        period : int, optional
            Period of the series (eg, 1 for annual, 4 for quarterly, etc). Must be
            used if x is not a pandas object or if the index of x does not have a
            frequency. Overrides default periodicity of x if x is a pandas object
            with a timeseries index.
        two_sided : bool, optional
            The moving average method used in filtering.
            If True (default), a centered moving average is computed using the
            filt. If False, the filter coefficients are for past values only.
        extrapolate_trend : int or 'freq', optional
            If set to > 0, the trend resulting from the convolution is
            linear least-squares extrapolated on both ends (or the single one
            if two_sided is False) considering this many (+1) closest points.
            If set to 'freq', use `freq` closest points. Setting this parameter
            results in no NaN values in trend or resid components.

        Returns
        -------
        X13Result
            A object with seasonal, trend, and resid attributes.

        """

    def fit(self) -> X13Result:
        """
        Estimate a trend component, multiple seasonal components, and a
        residual component.

        Returns
        -------
        X13Result
            Estimation results.
        """
        return X13Result(
            seasonal=None,
            trend=None,
            resid=None,
            observed=None,
        )

    def _select_order(self) -> None:
        """
        Select the order of seasonal and non-seasonal terms used in the ARIMA.
        """