# MX-Record-Checker
## Description

A Python utility to check the compatibility and health of MX records of email domains with their mail servers. This script performs an MX record lookup, resolves the MX domain to its IP, and tries to establish a Telnet connection on port 25 to verify if the mail server is active.

## Prerequisites

- Python 3.x
- dns.resolver
- socket
- telnetlib

You can install the required packages using pip:
```bash
pip install dnspython
```

## Usage

Clone this repository:
```bash
git clone https://github.com/kaiserm99/MX-Record-Checker.git
```

Navigate to the directory:
```bash
cd MX-Record-Checker
```

Run the script:
```bash
python check_mx_record.py
```
The script will then iterate through a list of popular email domains and check each domain's MX record, resolving it to an IP and attempting to connect via Telnet on port 25. The results will be printed on the console.

## Customization
You can add or remove domains from the popular_domains list in the script to check MX records for specific email domains.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT (https://choosealicense.com/licenses/mit/)

## Alpine inversion & fog model (`alpine_inversion.py`)

A second, unrelated utility that lives in this repo: it models the current
temperature inversion and fog situation between **Serfaus Dorf (~1427 m)** and
**Murmliwasser / Komperdell (~1980 m)** using the free
[Open-Meteo](https://open-meteo.com/) API. No API key and no third-party
packages are needed.

```bash
python alpine_inversion.py                 # text report, next 24 h
python alpine_inversion.py --hours 36      # longer time line
python alpine_inversion.py --model icon_d2 # force the 2 km ICON-D2 model
python alpine_inversion.py --json          # machine-readable output
python alpine_inversion.py --html chart.html # interactive diagram across the day
python alpine_inversion.py \
    --lower "Serfaus Dorf:47.0403:10.6031:1427" \
    --upper "Murmliwasser:47.0399:10.5628:1980"
```

What it computes:

- **Site-to-site lapse rate** (°C / 100 m) between the lower and upper site.
  A positive value means the mountain is warmer than the valley, i.e. an
  inversion. Standard atmosphere is about -0.65.
- **Potential-temperature gradient** as a stability measure.
- **Free-air inversion layers** from the model's pressure levels
  (850 / 800 / 700 hPa above the valley floor).
- **Fog score (0–10)** per site from dew-point spread, humidity, visibility,
  low cloud and wind, with a category (no fog / mist possible / fog likely).
- **Cloud base (LCL)** above the village and the **moist layer** (base and top
  of any RH ≥ 85 % layer), which tells whether the upper site sits above a
  sea of fog.
- A **time line** for the past 6 h and the next N hours showing when the
  inversion forms or breaks and how the fog risk evolves.
- With `--html FILE`, an **interactive diagram** (altitude cross-section with
  the cloud deck, temperatures, lapse rate and fog score across the day) built
  from `inversion_chart.template.html`.

Note: the 2 m values are model data downscaled to the given elevation, not
station observations, so treat the output as a model estimate.

## Hike forecast along a GPX track (`hike_forecast.py`)

Walks a GPX track with a Naismith/Langmuir time model, samples it every few
hundred metres and asks Open-Meteo for temperature, feels-like, humidity,
wind, gusts, cloud, visibility, precipitation and fog risk at each point,
interpolated to the estimated time of arrival.

```bash
python hike_forecast.py routes/quellenweg.gpx --start 10:00            # today, 10:00
python hike_forecast.py routes/quellenweg.gpx --start 2026-09-13T09:30
python hike_forecast.py routes/quellenweg.gpx --start 10:00 --duration 98   # force total minutes
python hike_forecast.py routes/quellenweg.gpx --start 10:00 --reverse      # walk uphill instead
python hike_forecast.py routes/quellenweg.gpx --start 10:00 --json
```

`routes/quellenweg.gpx` is the Quellenweg themed trail in Serfaus-Fiss-Ladis
(Schönjoch top station 2407 m down to the Komperdell station / Murmliwasser
at 1976 m, 5.6 km), taken from the public Komoot smart tour.
