"""forecast paketi — predict-then-optimize: CNN-LSTM bir-adim getiri tahmincisi.

Train-only fit edilir; tahmini state'e 'forecast' feature olarak eklenir (Faz V4).
"""
from forecast.forecaster import ReturnForecaster, build_forecast_feature

__all__ = ["ReturnForecaster", "build_forecast_feature"]
