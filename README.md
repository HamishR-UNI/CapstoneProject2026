# Plant Operator Web App

Browser dashboard for the ENGEN582 biomass CHP plant simulation (React frontend, FastAPI backend).
This folder is self-contained: the plant model files the web app imports sit at the top level,
and the web app itself is in `webapp/`. See `webapp/README.md` for details.

## Setup (once)

1. Python packages (from this folder):
   ```
   py -3.13 -m pip install -r requirements.txt
   ```
2. Build the frontend (needs Node.js 20+):
   ```
   cd webapp\frontend
   npm install
   npm run build
   ```

## Run

From this folder:
```
cd webapp\backend
py -3.13 -m uvicorn app:app --port 8000
```
then open http://127.0.0.1:8000. Building a simulation session takes one to two minutes.

## Layout

```
*.py                      plant model (boiler, turbine, heat pump, headers, controllers, economics)
excel_verify/python_side.py   steady-state closure used by the RTO controller
webapp/backend/           FastAPI server (adds this folder to the Python path)
webapp/frontend/          React + Vite source (build output in dist/ is not committed)
```
