"""Download the remaining L2 components, ingest, train and forecast in one run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from .cdse_archive import download_standard, save_json
    from .ice_forecast import ingest, train, forecast
except ImportError:
    from cdse_archive import download_standard, save_json
    from ice_forecast import ingest, train, forecast


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', required=True)
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-gb', type=float, default=40)
    p.add_argument('--validation-start', default='2026-05-15')
    p.add_argument('--test-start', default='2026-07-15')
    args = p.parse_args()
    if args.max_gb <= 0:
        p.error('max-gb must be positive')
    output, data = Path(args.out), Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pipeline-status.json'
    def status(stage, **extra):
        save_json(status_path, {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra})
    stage = 'download'
    try:
        catalog = json.loads(Path(args.catalogue).read_text(encoding='utf-8'))
        status(stage, limit_gb=args.max_gb)
        if not download_standard(catalog['products'], args.raw_dir, max_products=len(catalog['products']), max_gb=args.max_gb):
            status('download_limit_reached', message='Restart to continue or revise the volume limit; training has not run.')
            return 3
        stage = 'ingest'
        status(stage)
        observations = data / 'archive-nt-observations.csv'
        ingest([args.raw_dir], observations, timeliness='NT', catalogue_path=args.catalogue)
        stage = 'train'
        status(stage)
        train(observations, output / 'model', args.validation_start, args.test_start)
        stage = 'forecast'
        status(stage)
        issue = datetime.now(timezone.utc).isoformat()
        forecast(observations, output / 'model/model.joblib', issue, output / 'forecast.csv')
        status('complete', issue_time=issue, metrics='model/metrics.json', forecast='forecast.tif')
        return 0
    except Exception as exc:
        # Never record authentication response bodies or credentials.
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        status('failed', failed_stage=stage, error=message)
        print(f'Pipeline stopped in {stage}: {message}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
