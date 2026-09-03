# tree60-pipelines

Scheduled weather-raster jobs for [tree60weather.com](https://tree60weather.com).

These ran on Cloud Run at 2–8 GiB and were the bulk of a $291/month Google Cloud
bill. The data behind them (NOAA/NCEP public archives, dynamical.org Zarr) and the
rendering were always free — only the always-on container time was billed. They run
here on GitHub's runners instead and write their output to Cloudflare R2, which the
site serves from the edge.

## Layout

| Path | What |
|---|---|
| `r2_storage.py` | Drop-in stand-in for the slice of `google.cloud.storage` these modules use, backed by R2. Porting a module is a one-line import change. |
| `jobs/` | Entry points invoked by the workflows. |
| `*.py` (root) | Rendering modules, carried over from the Firebase functions codebase largely unmodified so the imagery does not change. |

## Configuration

Set as repository secrets:

- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

Output is written to the `tree60-data` bucket and served at
`https://tree60weather.com/api/data/<key>`.

## Running locally

```bash
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
./venv/bin/python jobs/temperature.py archive
```

Data sources are public; no NOAA or Cloudflare credentials are embedded in this
repository.
