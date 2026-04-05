"""
PriceDrive — Dash dashboard for vehicle selling price prediction (XGBoost).
Preprocessing matches project.ipynb: cleaned_dataset.csv + LabelEncoder on categoricals.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from predict import load_artifacts, predict_price

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "cleaned_dataset.csv"
SAMPLE_CAP = 12_000

_df = pd.read_csv(DATA_PATH)
_year_min = int(_df["year"].min())
_year_max = int(_df["year"].max())

# UI options and charts (string categoricals)
_raw = _df[
    ["make", "model", "trim", "body", "transmission", "sellingprice"]
].copy()
_makes = sorted(_raw["make"].astype(str).unique())
_model_by_make = {
    str(k): v
    for k, v in _raw.groupby("make")["model"]
    .agg(lambda s: sorted(s.astype(str).unique()))
    .items()
}
_trim_by_make_model = {
    (str(k[0]), str(k[1])): v
    for k, v in _raw.groupby(["make", "model"])["trim"]
    .agg(lambda s: sorted(s.astype(str).unique()))
    .items()
}
_bodies = sorted(_raw["body"].astype(str).unique())
_transmissions = sorted(_raw["transmission"].astype(str).unique())

_model, _encoders, _feature_columns = load_artifacts()

_importance = pd.DataFrame(
    {
        "feature": _feature_columns,
        "importance": _model.feature_importances_,
    }
).sort_values("importance", ascending=True)

_fig_importance = px.bar(
    _importance,
    x="importance",
    y="feature",
    orientation="h",
    title="Feature importance (XGBoost)",
    color="importance",
    color_continuous_scale="Viridis",
)
_fig_importance.update_layout(
    paper_bgcolor="#0f1419",
    plot_bgcolor="#1a2332",
    font=dict(color="#e6edf3"),
    title_font_size=16,
    margin=dict(l=120, r=40, t=48, b=40),
    coloraxis_showscale=False,
    yaxis=dict(title=""),
    xaxis=dict(title="Importance"),
)

_price_sample = _raw["sellingprice"].sample(
    min(SAMPLE_CAP, len(_raw)), random_state=42
)


def _dist_fig(vline: float | None = None) -> go.Figure:
    fig = px.histogram(
        _price_sample.to_frame(name="sellingprice"),
        x="sellingprice",
        nbins=60,
        title="Selling price distribution (sample of listings)",
        labels={"sellingprice": "Selling price (USD)", "count": "Count"},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        paper_bgcolor="#0f1419",
        plot_bgcolor="#1a2332",
        font=dict(color="#e6edf3"),
        title_font_size=16,
        margin=dict(l=48, r=24, t=48, b=48),
        showlegend=False,
        xaxis=dict(title="Selling price (USD)"),
        yaxis=dict(title="Count"),
    )
    if vline is not None:
        fig.add_vline(
            x=vline,
            line_color="#58a6ff",
            line_width=2,
            annotation_text="Your estimate",
            annotation_position="top",
        )
    return fig


_fig_dist = _dist_fig()


def input_style():
    return {
        "width": "100%",
        "padding": "10px 12px",
        "borderRadius": "8px",
        "border": "1px solid #2d3a4f",
        "backgroundColor": "#0f1419",
        "color": "#e6edf3",
        "boxSizing": "border-box",
    }


def dropdown_style():
    return {"color": "#0f1419"}


external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap"
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

CARD = {
    "backgroundColor": "#1a2332",
    "borderRadius": "12px",
    "padding": "20px 24px",
    "marginBottom": "16px",
    "border": "1px solid #2d3a4f",
}

app.layout = html.Div(
    style={
        "fontFamily": "'DM Sans', sans-serif",
        "background": "linear-gradient(160deg, #0b0f14 0%, #121a24 50%, #0f1419 100%)",
        "minHeight": "100vh",
        "color": "#e6edf3",
        "padding": "28px 32px 48px",
        "maxWidth": "1280px",
        "margin": "0 auto",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "PriceDrive",
                    style={
                        "margin": "0 0 8px 0",
                        "fontWeight": "700",
                        "fontSize": "2.1rem",
                        "letterSpacing": "-0.02em",
                    },
                ),
                html.P(
                    "Interactive selling price estimate from your notebook pipeline: "
                    "cleaned vehicle features, label-encoded categoricals, tuned XGBoost regressor.",
                    style={"margin": "0", "opacity": 0.85, "maxWidth": "720px"},
                ),
            ],
            style={"marginBottom": "28px"},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
                html.Div(
                    style=CARD,
                    children=[
                        html.H3(
                            "Vehicle details",
                            style={"marginTop": 0, "marginBottom": "16px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "14px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Label("Year"),
                                        dcc.Slider(
                                            id="year",
                                            min=_year_min,
                                            max=_year_max,
                                            step=1,
                                            value=max(
                                                _year_min,
                                                min(_year_max, 2015),
                                            ),
                                            marks={
                                                y: str(y)
                                                for y in sorted(
                                                    {
                                                        _year_min,
                                                        _year_max,
                                                        (_year_min + _year_max) // 2,
                                                    }
                                                    | set(
                                                        range(
                                                            _year_min,
                                                            _year_max + 1,
                                                            max(
                                                                1,
                                                                (_year_max - _year_min)
                                                                // 5,
                                                            ),
                                                        )
                                                    )
                                                )
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Vehicle age (at sale)"),
                                        dcc.Input(
                                            id="vehicle_age",
                                            type="number",
                                            value=3,
                                            min=0,
                                            max=40,
                                            step=0.5,
                                            style=input_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Condition"),
                                        dcc.Input(
                                            id="condition",
                                            type="number",
                                            value=45,
                                            min=1,
                                            max=50,
                                            step=0.5,
                                            style=input_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Odometer"),
                                        dcc.Input(
                                            id="odometer",
                                            type="number",
                                            value=45000,
                                            min=0,
                                            step=500,
                                            style=input_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Make"),
                                        dcc.Dropdown(
                                            id="make",
                                            options=[
                                                {"label": m, "value": m} for m in _makes
                                            ],
                                            value=_makes[0] if _makes else None,
                                            clearable=False,
                                            style=dropdown_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Model"),
                                        dcc.Dropdown(
                                            id="model",
                                            clearable=False,
                                            style=dropdown_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Trim"),
                                        dcc.Dropdown(
                                            id="trim",
                                            clearable=False,
                                            style=dropdown_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Body"),
                                        dcc.Dropdown(
                                            id="body",
                                            options=[
                                                {"label": b, "value": b}
                                                for b in _bodies
                                            ],
                                            value=_bodies[0] if _bodies else None,
                                            clearable=False,
                                            style=dropdown_style(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Transmission"),
                                        dcc.Dropdown(
                                            id="transmission",
                                            options=[
                                                {"label": t, "value": t}
                                                for t in _transmissions
                                            ],
                                            value=(
                                                "automatic"
                                                if "automatic" in _transmissions
                                                else _transmissions[0]
                                            ),
                                            clearable=False,
                                            style=dropdown_style(),
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Button(
                            "Estimate price",
                            id="predict-btn",
                            type="button",
                            n_clicks=0,
                            style={
                                "marginTop": "20px",
                                "padding": "12px 28px",
                                "fontSize": "1rem",
                                "fontWeight": "600",
                                "border": "none",
                                "borderRadius": "8px",
                                "cursor": "pointer",
                                "background": "linear-gradient(135deg, #3d8bfd, #1f6feb)",
                                "color": "#fff",
                                "position": "relative",
                                "zIndex": 2,
                            },
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            style=CARD,
                            children=[
                                html.H3(
                                    "Predicted selling price",
                                    style={"marginTop": 0},
                                ),
                                html.Div(
                                    id="price-out",
                                    style={
                                        "fontSize": "2.4rem",
                                        "fontWeight": "700",
                                        "color": "#58a6ff",
                                    },
                                    children="—",
                                ),
                                html.P(
                                    id="price-note",
                                    style={"opacity": 0.75, "fontSize": "0.9rem"},
                                    children="Enter details and click Estimate price.",
                                ),
                            ],
                        ),
                        html.Div(
                            style=CARD,
                            children=[
                                dcc.Graph(
                                    id="pred-chart",
                                    figure=_fig_dist,
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style=CARD,
            children=[
                dcc.Graph(
                    id="importance-chart",
                    figure=_fig_importance,
                    config={"displayModeBar": False},
                    style={"height": "420px"},
                ),
            ],
        ),
        # Browser writes here on each button click (avoids flaky n_clicks / timestamp on the server)
        dcc.Store(id="predict-gen", data=None),
    ],
)


@app.callback(
    Output("model", "options"),
    Output("model", "value"),
    Input("make", "value"),
)
def set_model_options(make):
    mk = str(make).strip() if make is not None else ""
    if not mk or mk not in _model_by_make:
        return [], None
    opts = _model_by_make[mk]
    return [{"label": m, "value": m} for m in opts], opts[0]


@app.callback(
    Output("trim", "options"),
    Output("trim", "value"),
    Input("make", "value"),
    Input("model", "value"),
)
def set_trim_options(make, model):
    mk = str(make).strip() if make is not None else ""
    mo = str(model).strip() if model is not None else ""
    if not mk or not mo:
        return [], None
    key = (mk, mo)
    if key not in _trim_by_make_model:
        return [], None
    opts = _trim_by_make_model[key]
    return [{"label": t, "value": t} for t in opts], opts[0]


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks === undefined || n_clicks === null || n_clicks < 1) {
            return window.dash_clientside.no_update;
        }
        return Date.now();
    }
    """,
    Output("predict-gen", "data"),
    Input("predict-btn", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("price-out", "children"),
    Output("price-note", "children"),
    Output("pred-chart", "figure"),
    Input("predict-gen", "data"),
    State("year", "value"),
    State("vehicle_age", "value"),
    State("condition", "value"),
    State("odometer", "value"),
    State("make", "value"),
    State("model", "value"),
    State("trim", "value"),
    State("body", "value"),
    State("transmission", "value"),
)
def run_predict(
    _gen,
    year,
    vehicle_age,
    condition,
    odometer,
    make,
    model,
    trim,
    body,
    transmission,
):
    if _gen is None:
        raise PreventUpdate

    year_f = _num(year)
    age_f = _num(vehicle_age)
    cond_f = _num(condition)
    odo_f = _num(odometer)

    missing = []
    if year_f is None:
        missing.append("year")
    if age_f is None:
        missing.append("vehicle age")
    if cond_f is None:
        missing.append("condition")
    if odo_f is None:
        missing.append("odometer")
    if not make or not model or not trim or not body or not transmission:
        missing.append("make/model/trim/body/transmission")

    if missing:
        return (
            "—",
            f"Please complete: {', '.join(missing)}.",
            _fig_dist,
        )

    try:
        price = predict_price(
            _model,
            _encoders,
            _feature_columns,
            year=year_f,
            vehicle_age=age_f,
            condition=cond_f,
            odometer=odo_f,
            make=str(make),
            model=str(model),
            trim=str(trim),
            body=str(body),
            transmission=str(transmission),
        )
    except Exception as e:  # noqa: BLE001
        return "—", f"Prediction error: {e}", _fig_dist

    price_fmt = f"${price:,.0f}"
    note = (
        "Point estimate from XGBoost; distribution shows a random sample of historical sales."
    )
    try:
        fig = _dist_fig(vline=float(price))
    except Exception:  # noqa: BLE001
        fig = _fig_dist
    return price_fmt, note, fig


def _port() -> int:
    return int(os.environ.get("PORT", "8050"))


if __name__ == "__main__":
    # Debug shows callback errors in the browser (set DASH_DEBUG=0 to turn off).
    _debug = os.environ.get("DASH_DEBUG", "1").strip().lower() not in ("0", "false", "no")
    # use_reloader=False avoids Werkzeug/watchdog ImportError on some Anaconda installs
    app.run(
        debug=_debug,
        host="0.0.0.0",
        port=_port(),
        use_reloader=False,
    )
