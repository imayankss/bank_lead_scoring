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
