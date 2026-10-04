import pandas as pd
import pytest

from src.train import WildfireModelTrainer


def test_single_sample_class_stays_in_training_split(tmp_path):
    feature_columns = [
        "elevation_m",
        "slope_pct",
        "dist_nearest_fire_m",
        "fires_within_5km",
        "fires_within_10km",
        "years_since_last_fire",
        "max_vpd",
        "max_wind_speed",
        "total_precip_annual",
    ]
    data = pd.DataFrame(
        [{**{column: index for column in feature_columns}, "is_high_risk": label}
         for index, label in enumerate([1, 0, 0, 0, 0])]
    )
    data_path = tmp_path / "features.parquet"
    data.to_parquet(data_path)

    trainer = WildfireModelTrainer.__new__(WildfireModelTrainer)
    trainer.data_path = str(data_path)

    with pytest.warns(UserWarning, match="only one sample"):
        X_train, X_test, y_train, y_test = trainer.load_and_prepare_data()

    assert 1 in y_train.to_numpy()
    assert 1 not in y_test.to_numpy()
    assert len(X_train) == 4
    assert len(X_test) == 1
