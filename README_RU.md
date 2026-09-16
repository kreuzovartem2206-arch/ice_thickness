# Ежедневная толщина морского льда Российской Арктики

Пайплайн строит ежедневный высокодетальный продукт толщины морского льда из спутниковых **Level-2** данных.

## Источники

- Sentinel-3A/B SRAL `SR_2_LAN_SI`, Level-2
- CryoSat-2 Level-2 NetCDF — локальное подключение

neXtSIM/CMEMS Level-4 в качестве входных данных **не используется**.

## Метод

Для каждой валидной L2-точки толщина пересчитывается самостоятельно:

```text
SIT = (rho_water * sea_ice_freeboard + snow_density * snow_depth)
      / (rho_water - ice_density)
```

где `rho_water = 1024 kg/m3`.

Для реальных Sentinel-3 файлов используются:

- `sea_ice_freeboard_20_ku`
- `radar_freeboard_20_ku` — сохраняется для анализа
- `snow_depth_sol1_20_ku` (fallback: `snow_depth_sol2_20_ku`)
- `snow_density_20_ku`
- `sea_ice_density_20_ku` (fallback: `ice_density_20_ku`)
- `sea_ice_concentration_20_ku`
- `surf_type_class_20_ku`

`sea_ice_thickness_20_ku` не используется для построения карты. Он сохраняется только для независимой проверки нашего расчёта; metadata содержит bias/MAE/RMSE/correlation.

## Пространственная схема

Исходные 20-Hz точки сохраняются без потери детализации. Ежедневный растр по умолчанию — **1 x 1 км**, но заполняются только ячейки, в которых есть реальные наблюдения. Дальняя пространственная интерполяция не выполняется.

Рекомендуемые режимы:

- 1000 м — максимальная детализация, более разреженная карта
- 2000 м — более устойчивый ежедневный продукт
- rolling window — 7 суток
- временной вес `w = exp(-age_days / 2)`

## Установка

Рекомендуется Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Авторизация Copernicus Data Space

```powershell
$env:CDSE_USERNAME="your@email.com"
$env:CDSE_PASSWORD="your_password"
```

Если включена 2FA:

```powershell
$env:CDSE_TOTP="123456"
```

## Запуск со скачиванием Sentinel-3

```powershell
python .\src\daily_multisat_l2_sit.py --date 2026-09-15 --days 7 --grid-m 1000 --s3-download --s3-raw-dir "C:\ice-thickness\data\sentinel3" --cs2-raw-dir "C:\ice-thickness\data\cryosat2" --output-dir "C:\ice-thickness\outputs"
```

Поиск CDSE заранее ограничен техническим сектором Российской Арктики: 65–90 N, 30–190 E, включая участок по другую сторону 180-го меридиана. Это предотвращает скачивание глобального архива Sentinel-3.

## Обработка уже скачанных Sentinel-3

```powershell
python .\src\daily_multisat_l2_sit.py --date 2026-09-15 --days 7 --grid-m 1000 --s3-raw-dir "C:\ice-thickness\data\sentinel3" --cs2-raw-dir "C:\ice-thickness\data\cryosat2" --output-dir "C:\ice-thickness\outputs"
```

В этом режиме `--s3-download` отсутствует, поэтому повторной загрузки нет.

## CryoSat-2

В текущей версии CryoSat-2 подключается локально: положите Level-2 `.nc` файлы в `--cs2-raw-dir`. Reader ищет распространённые алиасы freeboard, snow depth, snow density и ice density. Файл, в котором нет L2-полей, необходимых для собственного гидростатического расчёта, пропускается.

## Выходные данные

```text
multisat_l2_sit_observations_YYYYMMDD.csv.gz
multisat_l2_sit_cells_1000m_YYYYMMDD.csv
multisat_l2_sit_1000m_YYYYMMDD.tif
multisat_l2_observation_age_hours_1000m_YYYYMMDD.tif
multisat_l2_observation_count_1000m_YYYYMMDD.tif
multisat_l2_sit_std_1000m_YYYYMMDD.tif
multisat_l2_s3_count_1000m_YYYYMMDD.tif
multisat_l2_cs2_count_1000m_YYYYMMDD.tif
multisat_l2_sit_1000m_YYYYMMDD.png
multisat_l2_sit_metadata_YYYYMMDD.json
```

Основной GeoTIFF использует EPSG:3413.
