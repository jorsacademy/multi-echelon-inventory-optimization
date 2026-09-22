import numpy as np

from demand_inventory.forecasting import NaiveForecast, SeasonalForecast, rolling_mae


def test_naive_forecast_uses_recent_average():
    history = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    result = NaiveForecast(window=3).predict(history)
    assert np.isclose(result.mean, 6.0)
    assert result.std >= 1.0


def test_seasonal_forecast_recovers_weekly_pattern_better_than_naive():
    pattern = np.array([10, 20, 30, 40, 50, 60, 70], dtype=float)
    series = np.tile(pattern, 10)
    seasonal_mae = rolling_mae(series, SeasonalForecast(season_length=7), warmup=14)
    naive_mae = rolling_mae(series, NaiveForecast(window=7), warmup=14)
    assert seasonal_mae < naive_mae
