"""Research forecast of Sentinel-3-derived cell thickness seven calendar days ahead.

No spatial filling, future covariates, radar/ice-freeboard substitution or random split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import netCDF4
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from .cdse_archive import NAME, save_json
except ImportError:
    from cdse_archive import NAME, save_json

FIELDS = {'lat': 'lat_20_ku', 'lon': 'lon_20_ku', 'freeboard_m': 'sea_ice_freeboard_20_ku',
          'snow_m': 'snow_depth_sol1_20_ku', 'rho_s': 'snow_density_20_ku',
          'rho_i': 'sea_ice_density_20_ku', 'concentration_pct': 'sea_ice_concentration_20_ku',
          'surface': 'surf_type_class_20_ku'}
FEATURES = ['last_sit_m', 'mean7_m', 'mean_history_m', 'std_history_m', 'change_m_per_day',
            'age_days', 'n_days_history', 'snow_m', 'freeboard_m', 'concentration_pct',
            'x_km', 'y_km', 'season_sin', 'season_cos']


def utc(value):
    return pd.to_datetime(value, utc=True)


def read_product(path, grid_m, catalogue_item=None):
    match = NAME.fullmatch(path.name)
    if not match:
        raise ValueError('Not a Sentinel-3 SR_2_LAN_SI product')
    satellite, _, _, generated, timeliness, baseline = match.groups()
    nc_path = path / 'standard_measurement.nc'
    # netCDF4 applies scale/offset and missing-value masks automatically. Read
    # only the required variables: xarray's inspection of all fields is costly.
    with netCDF4.Dataset(nc_path) as ds:
        missing = set(FIELDS.values()) | {'time_20_ku'}
        missing -= set(ds.variables)
        if missing:
            raise ValueError('Incompatible product schema: ' + ', '.join(sorted(missing)))
        for field in ['freeboard_m', 'snow_m']:
            if getattr(ds[FIELDS[field]], 'units', '') != 'm':
                raise ValueError('Expected metres for ' + field)
        for field in ['rho_s', 'rho_i']:
            if getattr(ds[FIELDS[field]], 'units', '') not in ['kg/m^3', 'kg m-3', 'kg m**-3']:
                raise ValueError('Unexpected density units')
        if getattr(ds[FIELDS['concentration_pct']], 'units', '') not in ['percent', '%']:
            raise ValueError('Expected concentration in percent')
        surface = ds[FIELDS['surface']]
        meanings = getattr(surface, 'flag_meanings', '').split()
        values = np.asarray(getattr(surface, 'flag_values', []))
        if 'sea_ice' not in meanings or len(values) != len(meanings):
            raise ValueError('Unknown surface classification dictionary')
        sea_ice_value = values[meanings.index('sea_ice')]
        df = pd.DataFrame({k: np.ma.filled(np.ma.asarray(ds[v][:], dtype=float), np.nan) for k, v in FIELDS.items()})
        time_var = ds['time_20_ku']
        time_values = np.ma.asarray(time_var[:], dtype=float)
        good_time = ~np.ma.getmaskarray(time_values) & np.isfinite(np.ma.filled(time_values, np.nan))
        times = np.full(len(time_values), np.datetime64('NaT', 'us'), dtype='datetime64[us]')
        times[good_time] = np.asarray(netCDF4.num2date(time_values[good_time], units=time_var.units,
                                  calendar=getattr(time_var, 'calendar', 'standard'),
                                  only_use_cftime_datetimes=False), dtype='datetime64[us]')
        df['time'] = utc(times)
        processor = str(getattr(ds, 'source', 'unknown'))
    longitude = df.lon % 360
    good = (df.lat.between(65, 89.9) & longitude.between(30, 190) &
            df.surface.eq(sea_ice_value) &
            df.concentration_pct.between(15, 100) & df.freeboard_m.between(0, 3) &
            df.snow_m.between(0, 3) & df.rho_s.between(100, 600) & df.rho_i.between(800, 950))
    df = df.loc[good].copy()
    df['sit_m'] = (1024 * df.freeboard_m + df.rho_s * df.snow_m) / (1024 - df.rho_i)
    df = df[df.sit_m.between(0, 15) & df.time.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    x, y = Transformer.from_crs(4326, 3413, always_xy=True).transform(df.lon.to_numpy(), df.lat.to_numpy())
    df['ix'], df['iy'] = np.floor(x / grid_m).astype(int), np.floor(y / grid_m).astype(int)
    df['day'] = df.time.dt.floor('D')
    # Preserve product availability separately from observation date. No future publication leakage.
    metadata = catalogue_item
    if metadata is None and (path / 'download.json').exists():
        metadata = json.loads((path / 'download.json').read_text(encoding='utf-8'))
    publication = (metadata or {}).get('PublicationDate')
    available = utc(publication) if publication else utc(generated) + pd.Timedelta(days=1)
    records = []
    for (ix, iy, day), group in df.groupby(['ix', 'iy', 'day']):
        records.append({'ix': ix, 'iy': iy, 'day': day, 'obs_time': group.time.max(),
                        'available_at': max(available, group.time.max()),
                        'availability_verified': bool(publication), 'grid_m': grid_m,
                        'satellite': satellite, 'timeliness': timeliness, 'baseline': baseline,
                        'processor': processor, 'source': path.name, 'n_points': len(group),
                        **{v: float(group[v].median()) for v in ['sit_m', 'snow_m', 'freeboard_m', 'concentration_pct']}})
    return pd.DataFrame(records)


def ingest(raw_dirs, out, grid_m=25000, timeliness='ST', baseline='006', catalogue_path=None):
    paths = sorted({p for root in raw_dirs for p in Path(root).glob('*.SEN3')})
    selected = {}
    for p in paths:
        m = NAME.fullmatch(p.name)
        if not m or m.group(5) != timeliness or m.group(6) != baseline:
            continue
        key = m.group(1, 2, 3, 5, 6)
        if key not in selected or m.group(4) > NAME.fullmatch(selected[key].name).group(4):
            selected[key] = p
    lookup = {}
    if catalogue_path:
        lookup = {x['Name']: x for x in json.loads(Path(catalogue_path).read_text(encoding='utf-8'))['products']}
        selected = {key: path for key, path in selected.items() if path.name in lookup}
    frames, failures = [], []
    for i, path in enumerate(selected.values(), 1):
        if i == 1:
            print(f'Processing {len(selected)} catalogue-matched products', flush=True)
        try:
            f = read_product(path, grid_m, lookup.get(path.name))
            if not f.empty:
                frames.append(f)
        except Exception as exc:
            failures.append({'product': path.name, 'error': str(exc)})
        if i % 20 == 0 or i == len(selected):
            print(f'Ingest {i}/{len(selected)}; failures={len(failures)}', flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'discovered': len(paths), 'selected': len(selected), 'timeliness': timeliness,
              'baseline': baseline, 'grid_m': grid_m, 'failures': failures,
              'qa': 'unsmoothed corrected freeboard, sea_ice class, physical limits; no radar/smoothed fallback'}
    if frames:
        result = pd.concat(frames, ignore_index=True).sort_values(['day', 'ix', 'iy', 'source'])
        result.to_csv(out, index=False)
        report.update(rows=len(result), valid_points=int(result.n_points.sum()),
                      first_day=result.day.min().isoformat(), last_day=result.day.max().isoformat(),
                      cells=int(len(result[['ix', 'iy']].drop_duplicates())),
                      verified_availability_rows=int(result.availability_verified.sum()))
    save_json(out.with_suffix('.audit.json'), report)
    if not frames:
        raise ValueError('No valid observations; see audit JSON')
    print(json.dumps(report, ensure_ascii=True, default=str), flush=True)


def load_observations(path):
    df = pd.read_csv(path)
    for key in ['day', 'obs_time', 'available_at']:
        df[key] = utc(df[key])
    for key in ['grid_m', 'baseline', 'timeliness']:
        if df[key].nunique() != 1:
            raise ValueError('Cannot mix ' + key + ' in one model')
    if df.duplicated(['source', 'day', 'ix', 'iy']).any():
        raise ValueError('Duplicate product/cell/day records')
    if not np.isfinite(df[['sit_m', 'snow_m', 'freeboard_m', 'concentration_pct']]).all().all():
        raise ValueError('Non-finite observations')
    if (df.available_at < df.obs_time).any() or not df.sit_m.between(0, 15).all():
        raise ValueError('Invalid observation chronology or thickness')
    return df


def features_at(obs, issue_time, max_age_days=7, history_days=28):
    issue = utc(issue_time)
    past = obs[(obs.obs_time <= issue) & (obs.available_at <= issue) &
               (obs.obs_time > issue - pd.Timedelta(days=history_days))]
    rows = []
    for (ix, iy), g in past.groupby(['ix', 'iy']):
        # Equal weight for cell-days, not number of points or duplicate satellite segments.
        daily = g.groupby('day').agg(sit_m=('sit_m', 'median'), snow_m=('snow_m', 'median'),
                                    freeboard_m=('freeboard_m', 'median'),
                                    concentration_pct=('concentration_pct', 'median'), obs_time=('obs_time', 'max')).sort_index()
        last = daily.iloc[-1]
        age = (issue - last.obs_time).total_seconds() / 86400
        if age > max_age_days:
            continue
        week = daily[daily.obs_time > issue - pd.Timedelta(days=7)]
        first = daily.iloc[0]
        span = (last.obs_time - first.obs_time).total_seconds() / 86400
        grid_m = int(g.grid_m.iloc[0])
        target_day = issue.floor('D') + pd.Timedelta(days=7)
        angle = 2 * np.pi * target_day.dayofyear / 365.2425
        rows.append({'ix': ix, 'iy': iy, 'issue_time': issue, 'target_day': target_day,
                     'last_obs_time': last.obs_time, 'last_sit_m': last.sit_m,
                     'mean7_m': float(week.sit_m.mean()) if len(week) else float(last.sit_m),
                     'mean_history_m': float(daily.sit_m.mean()), 'std_history_m': float(daily.sit_m.std(ddof=0)),
                     'change_m_per_day': (last.sit_m - first.sit_m) / span if span > 0 else 0,
                     'age_days': age, 'n_days_history': len(daily), 'snow_m': last.snow_m,
                     'freeboard_m': last.freeboard_m, 'concentration_pct': last.concentration_pct,
                     'x_km': (ix + .5) * grid_m / 1000, 'y_km': (iy + .5) * grid_m / 1000,
                     'season_sin': np.sin(angle), 'season_cos': np.cos(angle)})
    return pd.DataFrame(rows)


def training_rows(obs, max_age_days=7, history_days=28):
    targets = obs.groupby(['ix', 'iy', 'day']).agg(target_sit_m=('sit_m', 'median'),
                    target_available_at=('available_at', 'max')).reset_index().rename(columns={'day': 'target_day'})
    rows = []
    for day in sorted(obs.day.unique()):
        issue = utc(day) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        f = features_at(obs, issue, max_age_days, history_days)
        if not f.empty:
            paired = f.merge(targets, on=['ix', 'iy', 'target_day'], how='inner', validate='one_to_one')
            if not paired.empty:
                rows.append(paired)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def split_rows(rows, val_start, test_start):
    val, test = utc(val_start), utc(test_start)
    if val >= test:
        raise ValueError('Validation must start before test')
    train = rows[(rows.issue_time < val) & (rows.target_day < val) & (rows.target_available_at < val)]
    validation = rows[(rows.issue_time >= val) & (rows.issue_time < test) &
                      (rows.target_day < test) & (rows.target_available_at < test)]
    holdout = rows[rows.issue_time >= test]
    return train.copy(), validation.copy(), holdout.copy()


def metrics(truth, prediction):
    error = np.asarray(prediction) - np.asarray(truth)
    return {'n': len(error), 'mae_m': float(np.abs(error).mean()),
            'rmse_m': float(np.sqrt(np.square(error).mean())), 'bias_m': float(error.mean())}


def train(obs_path, out, val_start, test_start, allow_estimated_availability=False):
    obs = load_observations(obs_path)
    if not obs.availability_verified.all() and not allow_estimated_availability:
        raise ValueError('Publication dates are missing. Supply catalogue metadata or explicitly enable research-only estimated availability.')
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    # NT is published weeks after acquisition. Its retrospective forecast explicitly
    # uses older observations, never pretends they were available on acquisition day.
    max_age_days, history_days = (45, 90) if obs.timeliness.iloc[0] == 'NT' else (7, 28)
    rows = training_rows(obs, max_age_days, history_days)
    if rows.empty:
        save_json(output / 'readiness.json', {'status': 'insufficient_data', 'paired_rows': 0,
                  'reason': 'No same-cell observations seven calendar days apart with available history.'})
        raise ValueError('No seven-day training pairs. Download a longer archive.')
    parts = split_rows(rows, val_start, test_start)
    counts = {name: {'rows': len(part), 'issue_days': int(part.issue_time.nunique())}
              for name, part in zip(['train', 'validation', 'test'], parts)}
    save_json(output / 'readiness.json', {'status': 'checking', 'counts': counts})
    if any(c['rows'] < 100 or c['issue_days'] < 14 for c in counts.values()):
        save_json(output / 'readiness.json', {'status': 'insufficient_data', 'counts': counts,
                  'reason': 'Each partition needs >=100 pairs and >=14 issue dates; this is only a minimum gate.'})
        raise ValueError('Insufficient independent dates/observations for train, validation and test')
    tr, va, te = parts
    # Fixed prototype settings. Validation chooses the candidate; test is used once for reporting.
    model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=.05,
                                         l2_regularization=10, early_stopping=False, random_state=42)
    model.fit(tr[FEATURES], tr.target_sit_m - tr.last_sit_m)
    def prediction(part):
        return np.clip(part.last_sit_m.to_numpy() + model.predict(part[FEATURES]), 0, 15)
    val_ml, val_base = metrics(va.target_sit_m, prediction(va)), metrics(va.target_sit_m, va.last_sit_m)
    selected = 'gradient_boosting' if val_ml['mae_m'] < val_base['mae_m'] else 'persistence'
    test_prediction = prediction(te) if selected == 'gradient_boosting' else te.last_sit_m.to_numpy()
    report = {'status': 'research_backtest_complete', 'target': 'Sentinel-3 L2-derived thickness, not independent ground truth',
              'horizon_days': 7, 'selected_on_validation': selected, 'counts': counts,
              'max_observation_age_days': max_age_days, 'history_days': history_days,
              'validation': {'gradient_boosting': val_ml, 'persistence': val_base},
              'test': {'selected': metrics(te.target_sit_m, test_prediction),
                       'persistence': metrics(te.target_sit_m, te.last_sit_m)},
              'availability': 'catalogue' if obs.availability_verified.all() else 'partly_assumed_production_plus_24h',
              'limitations': ['Cell-average forecast; no ice drift tracking or full Arctic coverage.',
                              'Only observed target cells are scored; no ice-free transition validation.',
                              'Correlated satellite errors; no validated uncertainty intervals.',
                              'Single chronological holdout; cross-season validation still required.']}
    # Deliberately keep the pre-validation fitted model, preserving a truthful untouched holdout.
    bundle = {'model': model, 'selected': selected, 'features': FEATURES, 'horizon_days': 7,
              'grid_m': int(obs.grid_m.iloc[0]), 'baseline': str(obs.baseline.iloc[0]),
              'timeliness': str(obs.timeliness.iloc[0]), 'test_start': str(test_start),
              'max_age_days': max_age_days, 'history_days': history_days,
              'trained_until': tr.target_available_at.max().isoformat(), 'report': report}
    joblib.dump(bundle, output / 'model.joblib')
    result = te[['ix', 'iy', 'issue_time', 'target_day', 'target_sit_m', 'last_sit_m', 'age_days']].copy()
    result['prediction_m'] = test_prediction
    result.to_csv(output / 'test_predictions.csv', index=False)
    save_json(output / 'metrics.json', report)
    print(json.dumps(report, indent=2))


def forecast(obs_path, model_path, issue_time, out):
    # Load only model artifacts created locally by this trusted program (joblib uses pickle).
    bundle = joblib.load(model_path)
    obs = load_observations(obs_path)
    for key in ['grid_m', 'baseline', 'timeliness']:
        if str(obs[key].iloc[0]) != str(bundle[key]):
            raise ValueError('Model/input mismatch: ' + key)
    if utc(issue_time) <= utc(bundle['trained_until']):
        raise ValueError('Forecast issue must follow training-data availability')
    f = features_at(obs, issue_time, bundle['max_age_days'], bundle['history_days'])
    if f.empty:
        raise ValueError('No eligible observations within the model age limit')
    f['forecast_sit_m'] = f.last_sit_m
    if bundle['selected'] == 'gradient_boosting':
        f['forecast_sit_m'] = np.clip(f.last_sit_m + bundle['model'].predict(f[FEATURES]), 0, 15)
    f['model'] = bundle['selected']
    f['status'] = 'experimental_unvalidated_for_navigation'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(out, index=False)
    transformer = Transformer.from_crs(3413, 4326, always_xy=True)
    lon, lat = transformer.transform(f.x_km.to_numpy() * 1000, f.y_km.to_numpy() * 1000)
    features = []
    for (_, row), x, y in zip(f.iterrows(), lon, lat):
        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [float(x), float(y)]},
                         'properties': {'forecast_sit_m': float(row.forecast_sit_m), 'age_days': float(row.age_days),
                                        'issue_time': row.issue_time.isoformat(), 'target_day': row.target_day.isoformat(),
                                        'grid_m': bundle['grid_m'], 'model': bundle['selected'], 'status': row.status}})
    save_json(out.with_suffix('.geojson'), {'type': 'FeatureCollection', 'features': features})
    import rasterio
    from rasterio.transform import from_origin
    grid = bundle['grid_m']
    xmin, xmax, ymin, ymax = int(f.ix.min()), int(f.ix.max()), int(f.iy.min()), int(f.iy.max())
    width, height = xmax - xmin + 1, ymax - ymin + 1
    if width * height > 20_000_000:
        raise ValueError('Raster extent too large; CSV/GeoJSON were saved')
    raster = np.full((height, width), -9999., dtype='float32')
    raster[ymax - f.iy.to_numpy(int), f.ix.to_numpy(int) - xmin] = f.forecast_sit_m.to_numpy()
    with rasterio.open(out.with_suffix('.tif'), 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype='float32', crs='EPSG:3413', nodata=-9999., compress='deflate',
                       transform=from_origin(xmin * grid, (ymax + 1) * grid, grid, grid)) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, 'Forecast sea-ice thickness (m)')
        dst.update_tags(horizon_days='7', issue_time=str(issue_time), model=bundle['selected'],
                        interpolation='none', status='research_prototype')
    save_json(out.with_suffix('.metadata.json'), {'horizon_days': 7, 'cells': len(f), 'grid_m': bundle['grid_m'],
              'target_definition': 'daily median on calendar day issue_date+7 (UTC)',
              'spatial_interpolation': 'none', 'uncertainty': 'not yet calibrated', 'report': bundle['report']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    ing = sub.add_parser('ingest')
    ing.add_argument('--raw-dir', action='append', required=True)
    ing.add_argument('--out', required=True)
    ing.add_argument('--grid-m', type=int, default=25000)
    ing.add_argument('--timeliness', choices=['NR', 'ST', 'NT'], default='ST')
    ing.add_argument('--baseline', default='006')
    ing.add_argument('--catalogue')
    tr = sub.add_parser('train')
    tr.add_argument('--observations', required=True)
    tr.add_argument('--out', required=True)
    tr.add_argument('--validation-start', required=True)
    tr.add_argument('--test-start', required=True)
    tr.add_argument('--allow-estimated-availability', action='store_true', help='Research hindcast only; not an operational backtest')
    pred = sub.add_parser('forecast')
    pred.add_argument('--observations', required=True)
    pred.add_argument('--model', required=True)
    pred.add_argument('--issue-time', required=True, help='Explicit UTC timestamp')
    pred.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        if args.command == 'ingest':
            if args.grid_m < 1000:
                p.error('Grid must be at least 1000 metres')
            ingest(args.raw_dir, args.out, args.grid_m, args.timeliness, args.baseline, args.catalogue)
        elif args.command == 'train':
            train(args.observations, args.out, args.validation_start, args.test_start, args.allow_estimated_availability)
        else:
            forecast(args.observations, args.model, args.issue_time, args.out)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
