# pricedrive
PriceDrive · Exploring 550K+ vehicle sales transactions to uncover pricing patterns, market trends, and predict selling prices using machine learning. Built with Pandas, Plotly Dash, and scikit-learn.

## Interactive dashboard (Dash)

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model and write `models/*.pkl`: `python train_model.py`
3. Run locally: `python app.py` then open `http://127.0.0.1:8050`

**Deploy online (example: [Render](https://render.com)):** connect this repo, use **Build command** `pip install -r requirements.txt && python train_model.py` and **Start command** `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 app:server` (see `render.yaml` / `Procfile`). The first build trains XGBoost on `dataset/cleaned_dataset.csv` and may take several minutes.

For a quick public URL from your machine, you can use a tunnel (e.g. `npx localtunnel --port 8050`) while `python app.py` is running; tunnels often show a one-time IP confirmation page on first visit.
