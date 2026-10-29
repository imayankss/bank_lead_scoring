# Lead Scoring Intelligence Web Dashboard

Modern Next.js dashboard for the lead scoring project.

## Data Contract

The frontend reads static JSON files from:

```text
web/public/data/
```

Regenerate them from the project root:

```bash
python3 scripts/run_export_web_data.py
```

The export writes strict JSON (no NaN), because the browser rejects a whole file over one NaN. `tests/test_web_exporter.py` checks the committed JSON against `reports/tables/`.

## Development

```bash
cd web
npm install
npm run dev
```

## Production Build

```bash
cd web
npm run build
```

The dashboard does not train models or duplicate ML logic. Python exports the artifacts; Next.js presents them.
