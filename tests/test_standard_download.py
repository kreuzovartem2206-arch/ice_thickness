import hashlib
import json

import netCDF4
import pytest

from src import cdse_archive as archive


@pytest.fixture
def product(tmp_path, monkeypatch):
    name = 'S3A_SR_2_LAN_SI_20260101T000000_20260101T001000_20260128T020000_0600_001_001______PS1_O_NT_006.SEN3'
    nc_path = tmp_path / 'fixture.nc'
    with netCDF4.Dataset(nc_path, 'w') as ds:
        ds.createDimension('n', 1)
        ds.createVariable('time_20_ku', 'f8', ('n',))[:] = [0]
        ds.product_name = name
    payload = nc_path.read_bytes()
    digest = hashlib.md5(payload).hexdigest()
    manifest = (f'<root><byteStream size="{len(payload)}"><fileLocation href="./standard_measurement.nc"/>'
                f'<checksum checksumName="MD5">{digest}</checksum></byteStream></root>').encode()
    contents = {'standard_measurement.nc': payload, 'xfdumanifest.xml': manifest}
    monkeypatch.setattr(archive, 'product_nodes', lambda s, item: {k: {'ContentLength': len(v)} for k, v in contents.items()})
    def fetch(s, auth, item, component, size, destination):
        destination.write_bytes(contents[component])
        return len(contents[component]), hashlib.md5(contents[component]).hexdigest()
    monkeypatch.setattr(archive, 'download_node', fetch)
    return {'Name': name, 'Id': 'test-id'}, contents


def test_component_download_and_resume(tmp_path, product):
    item, payload = product
    root = tmp_path / 'raw'
    assert archive.download_standard([item], root, 1, .1)
    complete = root / item['Name']
    assert (complete / 'standard_measurement.nc').read_bytes() == payload['standard_measurement.nc']
    assert not (complete / 'enhanced_measurement.nc').exists()
    assert json.loads((complete / 'download.json').read_text())['download_scope'] == 'standard_only'
    assert archive.download_standard([item], root, 1, .1)
    assert json.loads((root.parent / 'standard-download-progress.json').read_text())['completed_this_run'] == 0


def test_byte_budget_prevents_transfer(tmp_path, product):
    item, _ = product
    root = tmp_path / 'raw'
    assert not archive.download_standard([item], root, 1, 1e-12)
    assert not (root / item['Name']).exists()


def test_bad_checksum_never_marks_download_complete(tmp_path, monkeypatch, product):
    item, _ = product
    original = archive.download_node
    def bad(*args):
        n, digest = original(*args)
        return n, '0' * 32
    monkeypatch.setattr(archive, 'download_node', bad)
    root = tmp_path / 'raw'
    with pytest.raises(ValueError, match='MD5'):
        archive.download_standard([item], root, 1, .1)
    assert not (root / item['Name'] / 'download.json').exists()


def test_orchestrator_does_not_train_after_download_limit(tmp_path, monkeypatch):
    from src import run_archive_training as pipeline
    catalogue = tmp_path / 'catalogue.json'
    catalogue.write_text(json.dumps({'products': []}))
    out = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['run_archive_training', '--catalogue', str(catalogue),
                        '--raw-dir', str(tmp_path / 'raw'), '--data-dir', str(tmp_path / 'data'), '--out', str(out)])
    monkeypatch.setattr(pipeline, 'download_standard', lambda *a, **k: False)
    def forbidden(*a, **k):
        raise AssertionError('Training/ingestion must not start after a download limit')
    monkeypatch.setattr(pipeline, 'ingest', forbidden)
    monkeypatch.setattr(pipeline, 'train', forbidden)
    assert pipeline.main() == 3
    assert json.loads((out / 'pipeline-status.json').read_text())['stage'] == 'download_limit_reached'
