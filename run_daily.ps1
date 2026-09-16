param(
    [string]$Date = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd'),
    [int]$Days = 7,
    [int]$GridM = 1000,
    [string]$S3RawDir = 'data\raw\sentinel3',
    [string]$CS2RawDir = 'data\raw\cryosat2',
    [string]$OutputDir = 'outputs',
    [switch]$S3Download
)

$cmd = @(
  'src\daily_multisat_l2_sit.py',
  '--date', $Date,
  '--days', $Days,
  '--grid-m', $GridM,
  '--s3-raw-dir', $S3RawDir,
  '--cs2-raw-dir', $CS2RawDir,
  '--output-dir', $OutputDir
)

if ($S3Download) { $cmd += '--s3-download' }
python @cmd
