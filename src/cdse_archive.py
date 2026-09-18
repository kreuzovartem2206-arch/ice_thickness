"""Sentinel-3 L2 catalogue and resumable, bounded archive download. No saved secrets."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CATALOGUE = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
DOWNLOAD = 'https://download.dataspace.copernicus.eu/odata/v1/Products'
TOKEN = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
NAME = re.compile(r'^(S3[AB])_SR_2_LAN_SI_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{8}T\d{6}).*_(NR|ST|NT)_(\d{3})\.SEN3$')


def session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def catalogue(start, end, timeliness='ST', baseline='006'):
    # Bounding sector, not an administrative boundary. Point-level sea-ice QA follows.
    polygons = ['POLYGON((30 65,180 65,180 89.9,30 89.9,30 65))',
                'POLYGON((-180 65,-170 65,-170 89.9,-180 89.9,-180 65))']
    spatial = ' or '.join("OData.CSC.Intersects(area=geography'SRID=4326;" + p + "')" for p in polygons)
    filt = ("Collection/Name eq 'SENTINEL-3' and contains(Name,'SR_2_LAN_SI') and "
            f"contains(Name,'_{timeliness}_{baseline}.SEN3') and ({spatial}) and "
            f"ContentDate/Start ge {start}T00:00:00Z and ContentDate/Start lt {end}T00:00:00Z")
    params = {'$filter': filt, '$select': 'Id,Name,ContentDate,PublicationDate,ContentLength,Checksum',
              '$orderby': 'ContentDate/Start asc', '$top': 1000, '$count': 'true'}
    s, url, items = session(), CATALOGUE, []
    while url:
        if urlparse(url).hostname != 'catalogue.dataspace.copernicus.eu':
            raise ValueError('Unexpected catalogue pagination host')
        response = s.get(url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        items.extend(payload['value'])
        print(f"Catalogue: {len(items)} / {payload.get('@odata.count', '?')}", flush=True)
        url, params = payload.get('@odata.nextLink'), None
    # Same sensing interval may be reprocessed. Keep latest within ONE baseline/timeliness.
    unique = {}
    for item in items:
        match = NAME.match(item['Name'])
        if not match:
            continue
        satellite, first, last, generated, mode, base = match.groups()
        key = (satellite, first, last, mode, base)
        if key not in unique or generated > NAME.match(unique[key]['Name']).group(4):
            unique[key] = item
    return sorted(unique.values(), key=lambda p: (p['ContentDate']['Start'], p['Name']))


class Auth:
    def __init__(self):
        self.access = os.getenv('CDSE_ACCESS_TOKEN')
        self.refresh = os.getenv('CDSE_REFRESH_TOKEN')
        self.expires = time.time() + 60 if self.access else 0

    def token(self, force=False):
        if self.access and not force and time.time() < self.expires:
            return self.access
        data = {'client_id': 'cdse-public'}
        if self.refresh:
            data.update(grant_type='refresh_token', refresh_token=self.refresh)
        else:
            data.update(grant_type='password', username=os.getenv('CDSE_USERNAME') or input('CDSE email: ').strip(),
                        password=os.getenv('CDSE_PASSWORD') or getpass.getpass('CDSE password (hidden): '))
            totp = os.getenv('CDSE_TOTP') or input('2FA code (Enter if disabled): ').strip()
            if totp:
                data['totp'] = totp
        r = requests.post(TOKEN, data=data, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f'CDSE login failed (HTTP {r.status_code}); credentials are not logged.')
        body = r.json()
        self.access, self.refresh = body['access_token'], body.get('refresh_token')
        self.expires = time.time() + max(1, body.get('expires_in', 300) - 30)
        return self.access


def safe_members(zf, staging):
    root = staging.resolve()
    for info in zf.infolist():
        target = (root / info.filename).resolve()
        if not target.is_relative_to(root) or info.filename.startswith(('/', '\\')) or ':' in info.filename:
            raise ValueError('Unsafe path in downloaded archive')
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Symlinks are not supported in product archives')
        yield info


def download(items, raw_dir, max_products=20, max_gb=5):
    root = Path(raw_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    s, auth, count, transferred = session(), Auth(), 0, 0
    for item in items:
        if not NAME.fullmatch(item['Name']):
            raise ValueError('Unexpected product name')
        destination = root / item['Name']
        if (destination / 'download.json').exists() and (destination / 'standard_measurement.nc').exists():
            continue
        if destination.exists():
            raise RuntimeError(f'Unverified existing product: {destination}; use a separate download directory.')
        size = int(item.get('ContentLength', 0))
        if size <= 0:
            raise RuntimeError('Missing product size; refusing an unbounded transfer')
        if count >= max_products or transferred + size > max_gb * 1e9:
            break
        if shutil.disk_usage(root).free < size * 3 + 1e9:
            raise RuntimeError('Not enough free disk space for download and extraction')
        part = root / (item['Name'] + '.part')
        print(f'Download {count + 1}: {item["Name"]} ({size / 1e6:.1f} MB)', flush=True)
        for attempt in range(2):
            r = s.get(f'{DOWNLOAD}({item["Id"]})/$value', headers={'Authorization': 'Bearer ' + auth.token(force=attempt > 0)},
                      stream=True, timeout=(30, 180))
            if r.status_code == 401 and attempt == 0:
                r.close()
                continue
            r.raise_for_status()
            break
        received = 0
        hashes = {'md5': hashlib.md5(), 'sha256': hashlib.sha256()}
        with r, part.open('wb') as handle:
            for chunk in r.iter_content(1024 * 1024):
                received += len(chunk)
                if received > size + 1024 * 1024:
                    raise RuntimeError('Download exceeds catalogue size')
                handle.write(chunk)
                for digest in hashes.values():
                    digest.update(chunk)
        if received != size:
            raise RuntimeError('Download size mismatch; partial file retained for diagnosis')
        for checksum in item.get('Checksum', []):
            algorithm = checksum['Algorithm'].lower().replace('-', '')
            if algorithm in hashes and hashes[algorithm].hexdigest().lower() != checksum['Value'].lower():
                raise RuntimeError('Downloaded product checksum mismatch')
        staging = root / (item['Name'] + '.extracting')
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(part) as zf:
            members = list(safe_members(zf, staging))
            if sum(m.file_size for m in members) > max(size * 30, 2e9):
                raise ValueError('Unreasonable uncompressed archive size')
            bad = zf.testzip()
            if bad:
                raise ValueError('ZIP CRC failed')
            zf.extractall(staging, members=members)
        source = staging / item['Name']
        if not (source / 'standard_measurement.nc').exists():
            raise RuntimeError('Expected standard_measurement.nc missing')
        save_json(source / 'download.json', item)
        source.replace(destination)
        # Only these validated, task-created temporary files are removed.
        part.unlink()
        if not any(staging.iterdir()):
            staging.rmdir()
        count += 1
        transferred += received
    print(json.dumps({'downloaded_products': count, 'downloaded_gb': transferred / 1e9,
                      'remaining_products': sum(not (root / i['Name'] / 'download.json').exists() for i in items)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    cat = sub.add_parser('catalogue')
    cat.add_argument('--start', required=True)
    cat.add_argument('--end', required=True, help='Exclusive UTC date')
    cat.add_argument('--timeliness', choices=['NR', 'ST', 'NT'], default='ST')
    cat.add_argument('--baseline', default='006')
    cat.add_argument('--out', required=True)
    dl = sub.add_parser('download')
    dl.add_argument('--catalogue', required=True)
    dl.add_argument('--raw-dir', required=True)
    dl.add_argument('--max-products', type=int, default=20)
    dl.add_argument('--max-gb', type=float, default=5)
    args = p.parse_args()
    if args.command == 'catalogue':
        from datetime import date
        if date.fromisoformat(args.end) <= date.fromisoformat(args.start):
            p.error('End must follow start')
        items = catalogue(args.start, args.end, args.timeliness, args.baseline)
        save_json(args.out, {'start': args.start, 'end_exclusive': args.end, 'timeliness': args.timeliness,
                            'baseline': args.baseline, 'products': items,
                            'total_gb': sum(int(x.get('ContentLength', 0)) for x in items) / 1e9})
        print(f'{len(items)} products; {sum(int(x.get("ContentLength", 0)) for x in items) / 1e9:.2f} GB')
    else:
        if args.max_products < 1 or args.max_gb <= 0:
            p.error('Download limits must be positive')
        download(json.loads(Path(args.catalogue).read_text(encoding='utf-8'))['products'],
                 args.raw_dir, args.max_products, args.max_gb)


if __name__ == '__main__':
    main()
