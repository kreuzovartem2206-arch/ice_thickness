import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cdse_archive import safe_members
from src.ice_forecast import FEATURES, features_at, training_rows, split_rows, read_product, train, forecast


def observations(days=120, cells=10):
    rows = []
    for i, day in enumerate(pd.date_range('2025-01-01', periods=days, tz='UTC')):
        for cell in range(cells):
            rows.append({'ix': cell, 'iy': 2, 'day': day, 'obs_time': day + pd.Timedelta(hours=10),
                         'available_at': day + pd.Timedelta(hours=13), 'availability_verified': True,
                         'grid_m': 25000, 'baseline': '006', 'timeliness': 'NR',
                         'sit_m': 1 + .001 * i + .01 * cell, 'snow_m': .15,
                         'freeboard_m': .1, 'concentration_pct': 90,
                         'source': f'test-{i}-{cell}'})
    return pd.DataFrame(rows)


def test_future_observations_and_unpublished_measurements_cannot_change_features():
    obs = observations(12, 1)
    issue = pd.Timestamp('2025-01-05T23:59:59Z')
    before = features_at(obs, issue)[FEATURES]
    obs.loc[obs.obs_time > issue, 'sit_m'] = 14
    late = obs.iloc[[0]].copy()
    late['available_at'] = pd.Timestamp('2025-02-01T00:00Z')
    late['sit_m'] = 14
    after = features_at(pd.concat([obs, late]), issue)[FEATURES]
    pd.testing.assert_frame_equal(before, after)


def test_seven_calendar_days_not_seven_observation_rows():
    obs = observations(15, 1)
    obs = obs[~obs.day.dt.day.isin([3, 4, 5])]
    rows = training_rows(obs)
    assert (rows.target_day - rows.issue_time.dt.floor('D')).eq(pd.Timedelta(days=7)).all()
    assert rows.target_day.min() == pd.Timestamp('2025-01-08T00:00Z')


def test_split_purges_future_labels_and_their_publication():
    rows = training_rows(observations(120, 1))
    rows.loc[0, 'target_available_at'] = pd.Timestamp('2026-01-01T00:00Z')
    tr, va, te = split_rows(rows, '2025-03-01', '2025-04-01')
    assert tr.target_available_at.max() < pd.Timestamp('2025-03-01T00:00Z')
    assert tr.target_day.max() < va.issue_time.min()
    assert va.target_day.max() < te.issue_time.min()
    assert 0 not in tr.index


def test_stale_cells_do_not_get_forecasts():
    assert features_at(observations(2, 1), '2025-02-01T00:00Z').empty


def test_nt_delay_is_respected_without_rewriting_publication_dates():
    obs = observations(60, 1)
    obs['available_at'] = obs.obs_time + pd.Timedelta(days=28)
    obs['timeliness'] = 'NT'
    issue = pd.Timestamp('2025-02-15T23:59:59Z')
    assert features_at(obs, issue).empty
    result = features_at(obs, issue, max_age_days=45, history_days=90)
    assert len(result) == 1
    assert result.age_days.iloc[0] >= 28
    assert result.last_obs_time.iloc[0] <= issue - pd.Timedelta(days=28)


def test_unverified_publication_dates_block_operational_backtest(tmp_path):
    obs = observations(120)
    obs['availability_verified'] = False
    path = tmp_path / 'obs.csv'
    obs.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Publication dates'):
        train(path, tmp_path / 'model', '2025-03-01', '2025-04-01')


def test_short_realistic_archive_refuses_training(tmp_path):
    path = tmp_path / 'obs.csv'
    observations(2).to_csv(path, index=False)
    with pytest.raises(ValueError, match='No seven-day'):
        train(path, tmp_path / 'model', '2025-02-01', '2025-03-01')
    assert not (tmp_path / 'model/model.joblib').exists()


def test_archive_traversal_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('../escape.txt', 'bad')
    with zipfile.ZipFile(buf) as z, pytest.raises(ValueError, match='Unsafe'):
        list(safe_members(z, tmp_path))


def test_actual_schema_and_no_radar_fallback(tmp_path):
    name = 'S3A_SR_2_LAN_SI_20260101T000000_20260101T001000_20260101T020000_0600_001_001______PS1_O_NR_006.SEN3'
    folder = tmp_path / name
    folder.mkdir()
    values = {'lat_20_ku': [75., 75., 75.], 'lon_20_ku': [70., 70., 70.],
              'sea_ice_freeboard_20_ku': [.2, .2, .2], 'snow_depth_sol1_20_ku': [.18] * 3,
              'snow_density_20_ku': [320.] * 3, 'sea_ice_density_20_ku': [915.] * 3,
              'sea_ice_concentration_20_ku': [90.] * 3, 'surf_type_class_20_ku': [1, 1, 2],
              'sea_ice_interp_flag_20_ku': [0, 1, 0]}
    ds = xr.Dataset({k: ('n', v) for k, v in values.items()})
    ds['time_20_ku'] = ('n', pd.date_range('2026-01-01', periods=3, freq='s').values)
    for k in ['sea_ice_freeboard_20_ku', 'snow_depth_sol1_20_ku']:
        ds[k].attrs['units'] = 'm'
    for k in ['snow_density_20_ku', 'sea_ice_density_20_ku']:
        ds[k].attrs['units'] = 'kg/m^3'
    ds['sea_ice_concentration_20_ku'].attrs['units'] = 'percent'
    ds['surf_type_class_20_ku'].attrs.update(flag_values=[0, 1, 2, 3], flag_meanings='open_ocean sea_ice lead unclassified')
    path = folder / 'standard_measurement.nc'
    ds.to_netcdf(path)
    result = read_product(folder, 25000)
    # sea_ice_interp_flag belongs to the separate smoothed branch, not the
    # unsmoothed corrected freeboard used here (it is entirely missing in NR).
    assert result.n_points.sum() == 2
    assert np.isclose(result.sit_m.iloc[0], (1024 * .2 + 320 * .18) / (1024 - 915))
    ds = ds.rename({'sea_ice_freeboard_20_ku': 'radar_freeboard_20_ku'})
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match='Incompatible'):
        read_product(folder, 25000)


def test_end_to_end_synthetic_smoke_only(tmp_path):
    path = tmp_path / 'obs.csv'
    observations(140).to_csv(path, index=False)
    train(path, tmp_path / 'model', '2025-03-01', '2025-04-01')
    forecast(path, tmp_path / 'model/model.joblib', '2025-05-20T23:59:59Z', tmp_path / 'forecast.csv')
    result = pd.read_csv(tmp_path / 'forecast.csv')
    assert len(result) == 10
    assert result.forecast_sit_m.between(0, 15).all()
    assert pd.to_datetime(result.target_day, utc=True).eq(pd.Timestamp('2025-05-27T00:00Z')).all()
    report = json.loads((tmp_path / 'model/metrics.json').read_text())
    assert report['test']['selected']['n'] > 0
