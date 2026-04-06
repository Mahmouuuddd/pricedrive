import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ============================================================
# LOAD DATA & MODEL
# ============================================================
df = pd.read_csv(r'dataset\cleaned_dataset.csv')

# Ensure correct types
df['year']         = df['year'].astype(int)
df['condition']    = df['condition'].astype(float)
df['odometer']     = df['odometer'].astype(float)
df['sellingprice'] = df['sellingprice'].astype(float)
df['vehicle_age']  = df['vehicle_age'].astype(int)

# Load ML artifacts
model     = joblib.load(r'models\xgb_pricedrive.pkl')
encoders  = joblib.load(r'models\label_encoders.pkl')
feat_cols = joblib.load(r'models\feature_columns.pkl')

# Unique values for dropdowns
makes         = sorted(df['make'].dropna().unique().tolist())
body_types    = sorted(df[df['body'] != 'unknown']['body'].dropna().unique().tolist())
transmissions = sorted(df['transmission'].dropna().unique().tolist())

# ============================================================
# APP INIT
# ============================================================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server  # for Render deployment

# ============================================================
# THEME
# ============================================================
BG_PRIMARY   = "#0b0f14"
BG_SECONDARY = "#121a24"
BG_CARD      = "#1a2332"
BG_INPUT     = "#0f1419"
BORDER       = "#2d3a4f"
TEXT_MAIN    = "#e6edf3"
TEXT_MAIN2   = "#010f1b"
TEXT_MUTED   = "#8b949e"

ACCENT_BLUE  = "#3d8bfd"
ACCENT_SKY   = "#58a6ff"
ACCENT_RED   = "#ff7b72"
ACCENT_LAV   = "#a5b4fc"
ACCENT_CYAN  = "#79c0ff"

PLOT_TEMPLATE = "plotly_dark"

# ============================================================
# STYLES
# ============================================================
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0, "left": 0, "bottom": 0,
    "width": "240px",
    "padding": "32px 24px",
    "backgroundColor": BG_INPUT,
    "color": TEXT_MAIN,
    "borderRight": f"1px solid {BORDER}",
    "zIndex": 100,
}

CONTENT_STYLE = {
    "marginLeft": "260px",
    "padding": "32px",
    "background": f"linear-gradient(160deg, {BG_PRIMARY} 0%, {BG_SECONDARY} 50%, {BG_INPUT} 100%)",
    "minHeight": "100vh",
    "color": TEXT_MAIN,
}

CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "borderRadius": "14px",
    "boxShadow": "0 4px 18px rgba(0,0,0,0.20)",
    "border": f"1px solid {BORDER}",
    "padding": "20px",
    "marginBottom": "20px",
}

INPUT_STYLE = {
    "backgroundColor": BG_INPUT,
    "color": TEXT_MAIN,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "minHeight": "38px",
}
INPUT_STYLE_DROPDOWN = {
    "backgroundColor": BG_INPUT,
    "color": TEXT_MAIN2,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "minHeight": "38px",
}

RESULT_BOX_STYLE = {
    "borderRadius": "12px",
    "padding": "24px",
    "textAlign": "center",
    "backgroundColor": BG_INPUT,
    "border": f"2px dashed {BORDER}",
    "minHeight": "140px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "marginBottom": "16px",
}


GLOBAL_INLINE_CSS = f"""
:root {{
    --pd-bg-primary: {BG_PRIMARY};
    --pd-bg-secondary: {BG_SECONDARY};
    --pd-bg-card: {BG_CARD};
    --pd-bg-input: {BG_INPUT};
    --pd-border: {BORDER};
    --pd-text-main: {TEXT_MAIN};
    --pd-text-muted: {TEXT_MUTED};
    --pd-accent: {ACCENT_SKY};
    --pd-accent-strong: {ACCENT_BLUE};
    --pd-danger: {ACCENT_RED};
}}

/* Native inputs from dbc.Input */
.form-control,
.form-select,
input,
textarea,
select {{
    background-color: var(--pd-bg-input) !important;
    color: var(--pd-text-main) !important;
    border: 1px solid var(--pd-border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}

.form-control::placeholder,
input::placeholder,
textarea::placeholder {{
    color: var(--pd-text-muted) !important;
    opacity: 1 !important;
}}

.form-control:focus,
.form-select:focus,
input:focus,
textarea:focus,
select:focus {{
    background-color: var(--pd-bg-input) !important;
    color: var(--pd-text-main) !important;
    border-color: var(--pd-accent) !important;
    box-shadow: 0 0 0 0.2rem rgba(88, 166, 255, 0.15) !important;
}}

/* Dash Dropdown (react-select) */
.Select-control,
.Select-menu-outer,
.Select-menu,
.Select-value,
.Select-placeholder,
.Select-input > input,
.Select-arrow-zone,
.Select-clear-zone,
.Select-option,
.VirtualizedSelectOption,
.VirtualizedSelectFocusedOption {{
    background-color: var(--pd-bg-input) !important;
    color: var(--pd-text-main) !important;
}}

.Select-control {{
    border: 1px solid var(--pd-border) !important;
    border-radius: 8px !important;
    min-height: 38px !important;
    background-image: none !important;
}}

.is-focused:not(.is-open) > .Select-control,
.is-open > .Select-control,
.Select-control:hover {{
    border-color: var(--pd-accent) !important;
    box-shadow: 0 0 0 0.2rem rgba(88, 166, 255, 0.15) !important;
}}

.Select-placeholder {{
    color: var(--pd-text-muted) !important;
}}

.Select--single > .Select-control .Select-value,
.Select-value-label,
.has-value.Select--single > .Select-control .Select-value .Select-value-label {{
    color: var(--pd-text-main) !important;
}}

.Select-input > input {{
    color: var(--pd-text-main) !important;
}}

.Select-menu-outer {{
    border: 1px solid var(--pd-border) !important;
    border-top: none !important;
}}

.Select-option,
.VirtualizedSelectOption {{
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}}

.Select-option.is-focused,
.VirtualizedSelectFocusedOption {{
    background-color: #1f2a3a !important;
    color: var(--pd-text-main) !important;
}}

.Select-option.is-selected {{
    background-color: var(--pd-accent-strong) !important;
    color: white !important;
}}

.Select-arrow {{
    border-top-color: var(--pd-text-muted) !important;
}}

.is-open .Select-arrow {{
    border-bottom-color: var(--pd-text-muted) !important;
    border-top-color: transparent !important;
}}

.Select-clear-zone,
.Select-arrow-zone {{
    color: var(--pd-text-muted) !important;
}}

/* Number input spinners and disabled states */
input::-webkit-outer-spin-button,
input::-webkit-inner-spin-button {{
    filter: invert(0.85);
}}

input[disabled],
.form-control[disabled],
.Select.is-disabled > .Select-control {{
    opacity: 0.65 !important;
    background-color: #111822 !important;
}}

/* Keep graph modebar readable */
.modebar {{
    background: transparent !important;
}}
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>PriceDrive</title>
        {{%favicon%}}
        {{%css%}}
        <style>{GLOBAL_INLINE_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""




def style_figure(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MAIN),
        title_font=dict(size=18),
        margin=dict(l=40, r=30, t=60, b=40),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.10)"
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.10)"
    )
    return fig


def kpi_card(title, value, color=ACCENT_SKY):
    return html.Div([
        html.P(title, style={"color": TEXT_MUTED, "marginBottom": "6px", "fontSize": "13px"}),
        html.H4(value, style={"color": color, "fontWeight": "700", "margin": 0, "fontSize": "1.75rem"}),
    ], style={**CARD_STYLE, "borderLeft": f"4px solid {color}", "textAlign": "center"})


# ============================================================
# SIDEBAR
# ============================================================
sidebar = html.Div([
    html.Div([
        html.H3("PriceDrive", style={"color": TEXT_MAIN, "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Data in the driver's seat", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "32px"}),
    ]),
    dbc.Nav([
        dbc.NavLink("  Overview",        href="/",          active="exact",
                    style={"color": TEXT_MAIN, "marginBottom": "8px", "borderRadius": "10px", "padding": "12px 16px"}),
        dbc.NavLink("  Sales Trends",    href="/trends",    active="exact",
                    style={"color": TEXT_MAIN, "marginBottom": "8px", "borderRadius": "10px", "padding": "12px 16px"}),
        dbc.NavLink("  Market Analysis", href="/market",    active="exact",
                    style={"color": TEXT_MAIN, "marginBottom": "8px", "borderRadius": "10px", "padding": "12px 16px"}),
        dbc.NavLink("  Price Predictor", href="/predictor", active="exact",
                    style={"color": TEXT_MAIN, "marginBottom": "8px", "borderRadius": "10px", "padding": "12px 16px"}),
    ], vertical=True, pills=True),
], style=SIDEBAR_STYLE)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
def overview_layout():
    total_sales = f"{len(df):,}"
    avg_price   = f"${df['sellingprice'].mean():,.0f}"
    avg_cond    = f"{df['condition'].mean():.1f} / 50"
    avg_age     = f"{df['vehicle_age'].mean():.1f} yrs"
    top_make    = df['make'].value_counts().idxmax().title()
    top_body    = df['body'].value_counts().idxmax().title()

    # Condition distribution
    fig_cond = px.histogram(
        df, x='condition', nbins=50,
        title='Condition Rating Distribution',
        labels={'condition': 'Condition Score'},
        color_discrete_sequence=[ACCENT_SKY],
    )
    fig_cond.update_layout(bargap=0.05, height=320)
    style_figure(fig_cond)

    # Price distribution
    fig_price = px.histogram(
        df, x='sellingprice', nbins=80,
        title='Selling Price Distribution',
        labels={'sellingprice': 'Selling Price ($)'},
        color_discrete_sequence=[ACCENT_BLUE],
    )
    fig_price.update_layout(bargap=0.05, height=320)
    style_figure(fig_price)

    # Top 10 makes by volume
    top_makes = df['make'].value_counts().head(10).reset_index()
    top_makes.columns = ['make', 'count']
    fig_makes = px.bar(
        top_makes, x='count', y='make', orientation='h',
        title='Top 10 Makes by Sales Volume',
        labels={'count': 'Number of Sales', 'make': 'Make'},
        color='count', color_continuous_scale='Blues',
    )
    fig_makes.update_layout(
        height=360,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed'),
    )
    style_figure(fig_makes)

    # Odometer vs selling price scatter
    sample = df.sample(n=min(15_000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x='odometer', y='sellingprice',
        color='vehicle_age', color_continuous_scale='Viridis_r',
        opacity=0.4,
        title='Odometer vs Selling Price (colored by vehicle age)',
        labels={
            'odometer': 'Odometer (miles)',
            'sellingprice': 'Selling Price ($)',
            'vehicle_age': 'Vehicle Age (yrs)',
        }
    )
    fig_scatter.update_layout(height=360)
    
    style_figure(fig_scatter)

    return html.Div([
        html.H4("Dashboard Overview", style={"fontWeight": "700", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(kpi_card("Total Sales",     total_sales, ACCENT_SKY),  width=2),
            dbc.Col(kpi_card("Avg Price",       avg_price,   ACCENT_BLUE), width=2),
            dbc.Col(kpi_card("Avg Condition",   avg_cond,    "#f0b36b"),   width=2),
            dbc.Col(kpi_card("Avg Vehicle Age", avg_age,     ACCENT_LAV),  width=2),
            dbc.Col(kpi_card("Top Make",        top_make,    ACCENT_RED),  width=2),
            dbc.Col(kpi_card("Top Body Type",   top_body,    ACCENT_CYAN), width=2),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_cond)],    style=CARD_STYLE), width=6),
            dbc.Col(html.Div([dcc.Graph(figure=fig_price)],   style=CARD_STYLE), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_makes)],   style=CARD_STYLE), width=5),
            dbc.Col(html.Div([dcc.Graph(figure=fig_scatter)], style=CARD_STYLE), width=7),
        ]),
    ])


# ============================================================
# PAGE 2 — SALES TRENDS
# ============================================================
def trends_layout():
    return html.Div([
        html.H4("Sales Trends", style={"fontWeight": "700", "marginBottom": "20px"}),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Make", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED}),
                    dcc.Dropdown(
                        id='trend-make-filter',
                        options=[{'label': 'All Makes', 'value': 'all'}] +
                                [{'label': m.title(), 'value': m} for m in makes],
                        value='all', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Filter by Body Type", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED}),
                    dcc.Dropdown(
                        id='trend-body-filter',
                        options=[{'label': 'All Body Types', 'value': 'all'}] +
                                [{'label': b.title(), 'value': b} for b in body_types],
                        value='all', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Filter by Transmission", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED}),
                    dcc.Dropdown(
                        id='trend-trans-filter',
                        options=[{'label': 'All', 'value': 'all'}] +
                                [{'label': t.title(), 'value': t} for t in transmissions],
                        value='all', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                ], width=4),
            ]),
        ], style=CARD_STYLE),
        html.Div([dcc.Graph(id='trend-age-price')],   style=CARD_STYLE),
        html.Div([dcc.Graph(id='trend-cond-price')],  style=CARD_STYLE),
        html.Div([dcc.Graph(id='trend-year-volume')], style=CARD_STYLE),
    ])


# ============================================================
# PAGE 3 — MARKET ANALYSIS
# ============================================================
def market_layout():
    body_counts = df[df['body'] != 'unknown']['body'].value_counts().head(10).reset_index()
    body_counts.columns = ['body', 'count']
    fig_donut = px.pie(
        body_counts, names='body', values='count',
        hole=0.45, title='Sales Distribution by Body Type',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(height=380, showlegend=False)
    style_figure(fig_donut)

    make_summary = (
        df.groupby('make')
        .agg(count=('sellingprice', 'count'),
             avg_price=('sellingprice', 'mean'),
             avg_condition=('condition', 'mean'))
        .reset_index()
        .query('count >= 200')
        .sort_values('count', ascending=False)
        .head(20)
    )
    fig_bubble = px.scatter(
        make_summary, x='avg_price', y='count',
        size='count', color='avg_condition', text='make',
        color_continuous_scale='RdYlGn',
        title='Top Makes: Sales Volume vs Average Price',
        labels={
            'avg_price': 'Avg Selling Price ($)',
            'count': 'Sales Volume',
            'avg_condition': 'Avg Condition',
        }
    )
    fig_bubble.update_traces(textposition='top center', textfont_size=9)
    fig_bubble.update_layout(height=420)
    style_figure(fig_bubble)

    body_stats = (
        df[df['body'] != 'unknown']
        .groupby('body')
        .agg(count=('sellingprice', 'count'),
             avg_price=('sellingprice', 'mean'))
        .reset_index()
        .sort_values('count', ascending=False)
        .head(10)
    )
    fig_body = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Volume by Body Type', 'Avg Price by Body Type')
    )
    fig_body.add_trace(
        go.Bar(x=body_stats['body'], y=body_stats['count'],
               marker_color=ACCENT_BLUE, name='Volume'),
        row=1, col=1
    )
    fig_body.add_trace(
        go.Bar(x=body_stats.sort_values('avg_price', ascending=False)['body'],
               y=body_stats.sort_values('avg_price', ascending=False)['avg_price'],
               marker_color=ACCENT_RED, name='Avg Price'),
        row=1, col=2
    )
    fig_body.update_layout(height=360, showlegend=False)
    fig_body.update_xaxes(tickangle=-35)
    style_figure(fig_body)

    df['condition_bin'] = pd.cut(
        df['condition'],
        bins=[0, 10, 20, 30, 40, 50],
        labels=['1-10', '11-20', '21-30', '31-40', '41-50']
    )
    

    return html.Div([
        html.H4("Market Analysis", style={"fontWeight": "700", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_donut)],  style=CARD_STYLE), width=5),
            dbc.Col(html.Div([dcc.Graph(figure=fig_bubble)], style=CARD_STYLE), width=7),
        ]),
        html.Div([dcc.Graph(figure=fig_body)], style=CARD_STYLE),
        
    ])


# ============================================================
# PAGE 4 — PRICE PREDICTOR
# ============================================================
def predictor_layout():
    return html.Div([
        html.H4("🤖 Vehicle Price Predictor",
                style={"fontWeight": "700", "marginBottom": "8px"}),
        html.P("Fill in the vehicle details to get an estimated selling price.",
               style={"color": TEXT_MUTED, "marginBottom": "20px"}),

        html.Div([
            dbc.Row([
                # Column 1 — Vehicle identity
                dbc.Col([
                    html.Label("Make *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dcc.Dropdown(
                        id='pred-make',
                        options=[{'label': m.title(), 'value': m} for m in makes],
                        placeholder='Select make...', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                    html.Div(id='err-make', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Model *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dcc.Dropdown(
                        id='pred-model',
                        placeholder='Select make first...',
                        clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                    html.Div(id='err-model', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Trim", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dcc.Dropdown(
                        id='pred-trim',
                        placeholder='Select model first...',
                        clearable=True,
                        style=INPUT_STYLE_DROPDOWN,
                    ),

                    html.Br(),
                    html.Label("Body Type *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dcc.Dropdown(
                        id='pred-body',
                        options=[{'label': b.title(), 'value': b} for b in body_types],
                        placeholder='Select body type...', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                    html.Div(id='err-body', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Transmission *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dcc.Dropdown(
                        id='pred-transmission',
                        options=[{'label': t.title(), 'value': t} for t in transmissions],
                        placeholder='Select transmission...', clearable=False,
                        style=INPUT_STYLE_DROPDOWN,
                    ),
                    html.Div(id='err-transmission', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),
                ], width=4),

                # Column 2 — Numeric inputs
                dbc.Col([
                    html.Label("Year *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dbc.Input(
                        id='pred-year', type='number',
                        placeholder='e.g. 2014', min=1982, max=2015, step=1,
                        style=INPUT_STYLE,
                    ),
                    html.Div(id='err-year', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Odometer (miles) *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dbc.Input(
                        id='pred-odometer', type='number',
                        placeholder='e.g. 35000', min=10, max=300000, step=1,
                        style=INPUT_STYLE,
                    ),
                    html.Div(id='err-odometer', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Condition (1–50) *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dbc.Input(
                        id='pred-condition', type='number',
                        placeholder='e.g. 35', min=1, max=50, step=1,
                        style=INPUT_STYLE,
                    ),
                    html.Div(id='err-condition', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.Label("Vehicle Age (years) *", style={"fontWeight": "600", "fontSize": "13px", "color": TEXT_MAIN}),
                    dbc.Input(
                        id='pred-age', type='number',
                        placeholder='e.g. 3', min=0, max=33, step=1,
                        style=INPUT_STYLE,
                    ),
                    html.Div(id='err-age', style={"color": ACCENT_RED, "fontSize": "12px", "marginTop": "4px"}),

                    html.Br(),
                    html.P("* Required fields",
                           style={"color": TEXT_MUTED, "fontSize": "12px"}),
                ], width=4),

                # Column 3 — Button + result
                dbc.Col([
                    dbc.Button(
                        "Predict Price 🚗", id='pred-btn',
                        color='primary', size='lg',
                        style={
                            "width": "100%",
                            "fontWeight": "700",
                            "marginBottom": "20px",
                            "background": f"linear-gradient(135deg, {ACCENT_BLUE}, #1f6feb)",
                            "border": "none",
                        }
                    ),
                    html.Div(id='pred-result', style=RESULT_BOX_STYLE),
                    html.Div(id='pred-note', style={
                        "fontSize": "12px",
                        "color": TEXT_MUTED,
                        "textAlign": "center",
                    }),
                ], width=4),
            ]),
        ], style=CARD_STYLE),

        html.Div([
            html.H6("What drives the predicted price?",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            dcc.Graph(id='feat-importance'),
        ], style=CARD_STYLE),
    ])


# ============================================================
# APP LAYOUT
# ============================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(id='page-content', style=CONTENT_STYLE),
])


# ============================================================
# CALLBACK — ROUTING
# ============================================================
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/trends':
        return trends_layout()
    elif pathname == '/market':
        return market_layout()
    elif pathname == '/predictor':
        return predictor_layout()
    return overview_layout()


# ============================================================
# CALLBACKS — TRENDS
# ============================================================
@app.callback(
    Output('trend-age-price',   'figure'),
    Output('trend-cond-price',  'figure'),
    Output('trend-year-volume', 'figure'),
    Input('trend-make-filter',  'value'),
    Input('trend-body-filter',  'value'),
    Input('trend-trans-filter', 'value'),
)
def update_trends(make_filter, body_filter, trans_filter):
    dff = df.copy()
    if make_filter != 'all':
        dff = dff[dff['make'] == make_filter]
    if body_filter != 'all':
        dff = dff[dff['body'] == body_filter]
    if trans_filter != 'all':
        dff = dff[dff['transmission'] == trans_filter]

    # Depreciation curve
    age_stats = (
        dff[dff['vehicle_age'] <= 20]
        .groupby('vehicle_age')['sellingprice']
        .agg(['mean', 'median', 'std', 'count'])
        .reset_index()
    )
    age_stats['se']       = age_stats['std'] / np.sqrt(age_stats['count'])
    age_stats['ci_upper'] = age_stats['mean'] + 1.96 * age_stats['se']
    age_stats['ci_lower'] = age_stats['mean'] - 1.96 * age_stats['se']

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=pd.concat([age_stats['vehicle_age'], age_stats['vehicle_age'][::-1]]),
        y=pd.concat([age_stats['ci_upper'], age_stats['ci_lower'][::-1]]),
        fill='toself', fillcolor='rgba(61, 139, 253, 0.15)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI'
    ))
    fig1.add_trace(go.Scatter(
        x=age_stats['vehicle_age'], y=age_stats['mean'],
        mode='lines+markers', name='Mean price',
        line=dict(color=ACCENT_BLUE, width=2.5)
    ))
    fig1.add_trace(go.Scatter(
        x=age_stats['vehicle_age'], y=age_stats['median'],
        mode='lines+markers', name='Median price',
        line=dict(color=ACCENT_RED, width=2.5, dash='dash')
    ))
    fig1.update_layout(
        title='Price Depreciation by Vehicle Age',
        xaxis_title='Vehicle Age (years)',
        yaxis_title='Selling Price ($)',
        hovermode='x unified', height=360
    )
    style_figure(fig1)

    # Condition vs avg price
    dff['condition_bin'] = pd.cut(
        dff['condition'],
        bins=[0, 10, 20, 30, 40, 50],
        labels=['1-10', '11-20', '21-30', '31-40', '41-50']
    )
    cond_price = (
        dff.groupby('condition_bin', observed=True)['sellingprice']
        .mean().reset_index()
    )
    fig2 = px.bar(
        cond_price, x='condition_bin', y='sellingprice',
        title='Average Selling Price by Condition Rating',
        labels={'condition_bin': 'Condition (binned)',
                'sellingprice': 'Avg Selling Price ($)'},
        color='sellingprice', color_continuous_scale='RdYlGn',
        category_orders={'condition_bin': ['1-10', '11-20', '21-30', '31-40', '41-50']}
    )
    fig2.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
    style_figure(fig2)

    # Sales volume by manufacturing year
    year_vol = dff.groupby('year').size().reset_index(name='count')
    fig3 = px.bar(
        year_vol, x='year', y='count',
        title='Sales Volume by Manufacturing Year',
        labels={'year': 'Year', 'count': 'Number of Sales'},
        color='count', color_continuous_scale='Blues'
    )
    fig3.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
    style_figure(fig3)

    return fig1, fig2, fig3


# ============================================================
# CALLBACKS — CASCADE DROPDOWNS
# ============================================================
@app.callback(Output('pred-model', 'options'), Input('pred-make', 'value'))
def update_models(make):
    if not make:
        return []
    models_list = sorted(df[df['make'] == make]['model'].dropna().unique().tolist())
    return [{'label': m.title(), 'value': m} for m in models_list]


@app.callback(
    Output('pred-trim', 'options'),
    Input('pred-make', 'value'),
    Input('pred-model', 'value')
)
def update_trims(make, model_val):
    if not make or not model_val:
        return []
    trims = sorted(
        df[(df['make'] == make) & (df['model'] == model_val)]['trim']
        .dropna().unique().tolist()
    )
    return [{'label': t, 'value': t} for t in trims]


# ============================================================
# CALLBACK — PREDICT + VALIDATE
# ============================================================
@app.callback(
    Output('pred-result',      'children'),
    Output('pred-note',        'children'),
    Output('feat-importance',  'figure'),
    Output('err-make',         'children'),
    Output('err-model',        'children'),
    Output('err-body',         'children'),
    Output('err-transmission', 'children'),
    Output('err-year',         'children'),
    Output('err-odometer',     'children'),
    Output('err-condition',    'children'),
    Output('err-age',          'children'),
    Input('pred-btn', 'n_clicks'),
    State('pred-make',         'value'),
    State('pred-model',        'value'),
    State('pred-trim',         'value'),
    State('pred-body',         'value'),
    State('pred-transmission', 'value'),
    State('pred-year',         'value'),
    State('pred-odometer',     'value'),
    State('pred-condition',    'value'),
    State('pred-age',          'value'),
    prevent_initial_call=True,
)
def predict_price(n_clicks, make, model_val, trim, body,
                  transmission, year, odometer, condition, age):

    # ── Validation ──────────────────────────────────────────
    err = {k: '' for k in ['make', 'model', 'body', 'transmission',
                           'year', 'odometer', 'condition', 'age']}
    has_error = False

    if not make:
        err['make'] = '⚠ Make is required';                  has_error = True
    if not model_val:
        err['model'] = '⚠ Model is required';                has_error = True
    if not body:
        err['body'] = '⚠ Body type is required';             has_error = True
    if not transmission:
        err['transmission'] = '⚠ Transmission is required';  has_error = True
    if year is None:
        err['year'] = '⚠ Year is required';                  has_error = True
    elif not (1982 <= int(year) <= 2015):
        err['year'] = '⚠ Must be between 1982–2015';         has_error = True
    if odometer is None:
        err['odometer'] = '⚠ Odometer is required';          has_error = True
    elif not (10 <= int(odometer) <= 300000):
        err['odometer'] = '⚠ Must be 10–300,000 miles';      has_error = True
    if condition is None:
        err['condition'] = '⚠ Condition is required';        has_error = True
    elif not (1 <= int(condition) <= 50):
        err['condition'] = '⚠ Must be between 1–50';         has_error = True
    if age is None:
        err['age'] = '⚠ Vehicle age is required';            has_error = True
    elif int(age) < 0:
        err['age'] = '⚠ Cannot be negative';                 has_error = True

    empty_fig = go.Figure()
    empty_fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MAIN),
        height=300,
        annotations=[dict(
            text="Fill in the form and click Predict",
            showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )]
    )

    if has_error:
        msg = html.P("⚠ Please fix the errors highlighted above.",
                     style={"color": ACCENT_RED, "fontWeight": "600"})
        return (msg, '', empty_fig,
                err['make'], err['model'], err['body'], err['transmission'],
                err['year'], err['odometer'], err['condition'], err['age'])

    # ── Build input row ──────────────────────────────────────
    trim_val = trim if trim else (
        df[(df['make'] == make) & (df['model'] == model_val)]['trim'].mode()
        .iloc[0] if len(df[(df['make'] == make) & (df['model'] == model_val)]) > 0
        else 'base'
    )

    row = {
        'year':         int(year),
        'make':         make,
        'model':        model_val,
        'trim':         trim_val,
        'body':         body,
        'transmission': transmission,
        'condition':    int(condition),
        'odometer':     int(odometer),
        'vehicle_age':  int(age),
    }
    input_df = pd.DataFrame([row])

    # ── Encode categoricals ──────────────────────────────────
    for col, le in encoders.items():
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            else:
                input_df[col] = 0

    # ── Align to training feature order ─────────────────────
    for col in feat_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feat_cols]

    # ── Predict ─────────────────────────────────────────────
    predicted = float(model.predict(input_df)[0])
    predicted = max(500.0, predicted)

    # Market context from similar vehicles
    similar = df[
        (df['make'] == make) &
        (df['body'] == body) &
        (df['vehicle_age'].between(int(age) - 1, int(age) + 1))
    ]['sellingprice']

    if len(similar) >= 5:
        mkt_low  = similar.quantile(0.25)
        mkt_high = similar.quantile(0.75)
        note_text = (f"Similar {make.title()} {body}s aged ~{age} yrs "
                     f"typically sell between ${mkt_low:,.0f} – ${mkt_high:,.0f}")
    else:
        note_text = "Not enough similar vehicles in dataset for market comparison."

    result = html.Div([
        html.P("Estimated Selling Price",
               style={"color": TEXT_MUTED, "marginBottom": "4px", "fontSize": "13px"}),
        html.H2(f"${predicted:,.0f}",
                style={"color": ACCENT_SKY, "fontWeight": "800", "margin": "4px 0"}),
        html.P(f"{make.title()} · {model_val.title()} · {int(year)} · {int(odometer):,} mi",
               style={"color": TEXT_MUTED, "fontSize": "12px", "margin": 0}),
    ])

    # ── Feature importance ───────────────────────────────────
    feat_imp = (
        pd.DataFrame({'feature': feat_cols,
                      'importance': model.feature_importances_})
        .sort_values('importance', ascending=True)
    )
    fig_imp = px.bar(
        feat_imp, x='importance', y='feature', orientation='h',
        title='Feature Importance — What drives the prediction',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance', color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
    style_figure(fig_imp)

    return (result, note_text, fig_imp,
            '', '', '', '', '', '', '', '')


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
