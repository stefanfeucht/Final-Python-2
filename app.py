# app.py - Dashboard completo para el entregable (Final.xlsx + Crypto_historical_data.csv.gz)
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ---------------------------
# Inicializar
# ---------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Final Python - Proyecto"

# ---------------------------
# Cargar datos (archivo en la raíz del repo)
# ---------------------------
EXCEL = "Final.xlsx"
CRYPTO_CSV = "Crypto_historical_data.csv.gz"

# read sheets safely
P_weeklyS = pd.read_excel(EXCEL, sheet_name="P WeeklyS")
R_weeklyS = pd.read_excel(EXCEL, sheet_name="R WeeklyS")
P_weeklyC = pd.read_excel(EXCEL, sheet_name="P WeeklyC")
R_weeklyC = pd.read_excel(EXCEL, sheet_name="R WeeklyC")

# arreglar nombre columna fecha si hace falta
def fix_date(df):
    # rename first col that looks like date
    for c in df.columns:
        if c.lower() in ("date","order date","fecha"):
            df = df.rename(columns={c:"Date"})
            break
    if "Date" not in df.columns:
        raise ValueError("No se encontró columna Date en df")
    if pd.api.types.is_numeric_dtype(df["Date"]):
        df["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Date"], "D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

P_weeklyS = fix_date(P_weeklyS); R_weeklyS = fix_date(R_weeklyS)
P_weeklyC = fix_date(P_weeklyC); R_weeklyC = fix_date(R_weeklyC)

# tickers lists
stocks = [c for c in P_weeklyS.columns if c != "Date"]
cryptos = [c for c in P_weeklyC.columns if c != "Date"]

# ---------------------------
# Utilidades de métricas de riesgo
# ---------------------------
def historical_var(series, level=0.95):
    q = np.percentile(series.dropna(), 100*(1-level))
    return q

def cvar_historical(series, level=0.95):
    s = series.dropna()
    var = historical_var(s, level)
    tail = s[s <= var]
    return tail.mean() if len(tail)>0 else var

def max_drawdown(series):
    # series de retornos -> convertir a wealth index (1+ret acumulado)
    w = (1+series).cumprod()
    running_max = w.cummax()
    drawdown = (w - running_max) / running_max
    return drawdown.min()

def sharpe_ann(series, rf=0.0, periods_per_year=52):
    s = series.dropna()
    if len(s)==0 or s.std(ddof=1)==0: return np.nan
    ann_ret = s.mean()*periods_per_year
    ann_vol = s.std(ddof=1)*np.sqrt(periods_per_year)
    return (ann_ret - rf)/ann_vol

# ---------------------------
# Texto descriptivo para las 6 empresas (1.a)
# ---------------------------
# Selecciono 6 empresas representativas (puedes cambiar)
six_companies = ["Apple","Microsoft","Amazon","Google","Meta","Nvidia"]
company_descr = {
    "Apple":"Consumer electronics and services (iPhone, Mac, App Store).",
    "Microsoft":"Software, cloud (Azure), productivity (Office), enterprise services.",
    "Amazon":"E-commerce, cloud (AWS), logistics and consumer services.",
    "Google":"Search, ads, cloud, Android, and consumer services (Alphabet).",
    "Meta":"Social networks (Facebook, Instagram), ads, virtual reality development.",
    "Nvidia":"GPUs for gaming, datacenter, AI and acceleration hardware."
}

# ---------------------------
# Layout
# ---------------------------
app.layout = dbc.Container([
    html.H1("Proyecto Final - Análisis Acciones y Criptos", className="mt-3"),
    html.P("Instrucciones: use los controles para seleccionar acciones/cripto, períodoy tipo de gráfico.", className="lead"),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="1. Exploración Acciones", value="tab1"),
        dcc.Tab(label="2. Análisis Retornos (3 años)", value="tab2"),
        dcc.Tab(label="3. Criptomonedas", value="tab3"),
    ]),

    html.Div(id="tabs-content", className="mt-3")
], fluid=True)

# ---------------------------
# Callbacks para contenido de tabs
# ---------------------------
@app.callback(Output("tabs-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab=="tab1":
        return tab1_layout()
    if tab=="tab2":
        return tab2_layout()
    return tab3_layout()

# ---------------------------
# TAB 1: Exploración acciones (1.a + 1.b)
# ---------------------------
def tab1_layout():
    return html.Div([
        html.H4("1.a — Breve descripción de 6 empresas"),
        html.Ul([html.Li([html.B(c+": "), company_descr.get(c,"-")]) for c in six_companies]),

        html.Hr(),
        html.H4("1.b — Gráfica de Precios / Retornos"),
        dbc.Row([
            dbc.Col([
                html.Label("Selecciona acciones:"),
                dcc.Dropdown(id="sel-stocks", options=[{"label":s,"value":s} for s in stocks],
                             value=six_companies, multi=True)
            ], md=5),
            dbc.Col([
                html.Label("Mostrar:"),
                dcc.Dropdown(id="sel-type", options=[
                    {"label":"Precio de cierre","value":"P"},
                    {"label":"Retorno semanal","value":"R"}], value="P", clearable=False)
            ], md=2),
            dbc.Col([
                html.Label("Rango de fechas:"),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=P_weeklyS["Date"].min().date(),
                    max_date_allowed=P_weeklyS["Date"].max().date(),
                    start_date=P_weeklyS["Date"].min().date(),
                    end_date=P_weeklyS["Date"].max().date()
                )
            ], md=5)
        ], className="mb-3"),
        dcc.Graph(id="stocks-graph", config={"scrollZoom":True})
    ])

@app.callback(
    Output("stocks-graph","figure"),
    [Input("sel-stocks","value"),
     Input("sel-type","value"),
     Input("date-range","start_date"),
     Input("date-range","end_date")]
)
def update_stocks_graph(selected_stocks, tipo, start_date, end_date):
    if not selected_stocks:
        selected_stocks = six_companies
    start = pd.to_datetime(start_date); end = pd.to_datetime(end_date)
    df = P_weeklyS if tipo=="P" else R_weeklyS
    dff = df[(df["Date"]>=start)&(df["Date"]<=end)]
    fig = go.Figure()
    for s in selected_stocks:
        if s in dff.columns:
            fig.add_trace(go.Scatter(x=dff["Date"], y=dff[s], mode="lines", name=s))
    fig.update_layout(template="plotly_white", title=f"{'Precios' if tipo=='P' else 'Retornos'} seleccionado(s)")
    return fig

# ---------------------------
# TAB 2: Analizar retornos últimos 3 años (2.a - 2.d)
# ---------------------------
def tab2_layout():
    # default date window: last 3 years from R_weeklyS max
    end = R_weeklyS["Date"].max().date()
    start = (R_weeklyS["Date"].max() - pd.DateOffset(years=3)).date()
    return html.Div([
        html.H4("2 — Análisis de Retornos (últimos 3 años)"),
        html.P("Seleccione acciones para comparar distribuciones y métricas."),

        dbc.Row([
            dbc.Col(dcc.Dropdown(id="sel-stocks-2", options=[{"label":s,"value":s} for s in stocks],
                                 value=[s for s in six_companies if s in stocks], multi=True), md=6),
            dbc.Col(html.Div([
                html.Label("Selector de stock para distribución individual:"),
                dcc.Dropdown(id="dist-stock", options=[{"label":s,"value":s} for s in stocks],
                             value=six_companies[0], clearable=False)
            ]), md=6)
        ]),
        dcc.Graph(id="dist-compare", className="mb-3"),
        dcc.Graph(id="hist-single", className="mb-3"),
        html.H5("Métricas calculadas (para los seleccionados)"),
        dash_table.DataTable(id="metrics-table", style_table={"overflowX":"auto"})
    ])

@app.callback(
    [Output("dist-compare","figure"),
     Output("hist-single","figure"),
     Output("metrics-table","data"),
     Output("metrics-table","columns")],
    [Input("sel-stocks-2","value"),
     Input("dist-stock","value")]
)
def update_analysis(selected, single):
    # prepare data of last 3 years
    end = R_weeklyS["Date"].max()
    start = end - pd.DateOffset(years=3)
    R3 = R_weeklyS[(R_weeklyS["Date"]>=start)&(R_weeklyS["Date"]<=end)].copy()
    # compare densities
    fig_cmp = go.Figure()
    metrics = []
    for s in (selected or []):
        if s not in R3.columns: continue
        series = R3[s].dropna()
        # kernel density via histogram smoothing
        fig_cmp.add_trace(go.Histogram(x=series, name=s, opacity=0.5, nbinsx=50))
        # metrics
        metrics.append({
            "Acción": s,
            "Media": round(series.mean(),6),
            "Std": round(series.std(),6),
            "Curtosis": round(kurtosis(series, fisher=False),6),
            "Skew": round(skew(series),6),
            "VaR95": round(historical_var(series,0.95),6),
            "VaR90": round(historical_var(series,0.90),6),
            "CVaR95": round(cvar_historical(series,0.95),6),
            "MaxDrawdown": round(max_drawdown(series),6),
            "SharpeAnn": round(sharpe_ann(series),6)
        })
    fig_cmp.update_layout(barmode="overlay", template="plotly_white", title="Comparación histograma de retornos (últimos 3 años)")
    # single histogram + density
    fig_single = go.Figure()
    if single in R3.columns:
        srs = R3[single].dropna()
        fig_single.add_trace(go.Histogram(x=srs, nbinsx=60, name="Histograma", opacity=0.7))
        xvals = np.linspace(srs.min(), srs.max(), 200)
        fig_single.add_trace(go.Scatter(x=xvals, y=norm.pdf(xvals, srs.mean(), srs.std())*len(srs)*(srs.max()-srs.min())/60,
                                        mode="lines", name="Normal approx"))
        fig_single.update_layout(template="plotly_white", title=f"Distribución de {single}")
    cols = [{"name":c,"id":c} for c in metrics[0].keys()] if metrics else []
    return fig_cmp, fig_single, metrics, cols

# ---------------------------
# TAB 3: Criptomonedas (3.a, 3.b, 3.c)
# ---------------------------
def tab3_layout():
    # default crypto for animation
    return html.Div([
        html.H4("3 — Criptomonedas"),
        html.P("Bollinger bands sobre retornos y animación de precios."),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="sel-crypto", options=[{"label":c,"value":c} for c in cryptos],
                                 value=cryptos[0], clearable=False), md=4),
            dbc.Col(dcc.Dropdown(id="sel-crypto-compare", options=[{"label":c,"value":c} for c in cryptos],
                                 value=cryptos[:3], multi=True), md=8)
        ]),
        dcc.Graph(id="bollinger-graph"),
        dcc.Graph(id="crypto-anim"),
        html.H5("Métricas (CVaR y MaxDrawdown) para criptos seleccionadas"),
        dash_table.DataTable(id="crypto-metrics", style_table={"overflowX":"auto"})
    ])

@app.callback(
    [Output("bollinger-graph","figure"),
     Output("crypto-anim","figure"),
     Output("crypto-metrics","data"),
     Output("crypto-metrics","columns")],
    [Input("sel-crypto","value"),
     Input("sel-crypto-compare","value")]
)
def update_crypto(crypto_single, cryptos_sel):
    # Bollinger on returns for single crypto
    r = R_weeklyC.set_index("Date")[crypto_single].dropna()
    window=20
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std()
    upper = roll_mean + 2*roll_std
    lower = roll_mean - 2*roll_std
    fig_b = go.Figure()
    fig_b.add_trace(go.Line(x=r.index, y=r, name="Retorno"))
    fig_b.add_trace(go.Line(x=roll_mean.index, y=roll_mean, name="MA(20)"))
    fig_b.add_trace(go.Line(x=upper.index, y=upper, name="+2σ", line=dict(dash="dot")))
    fig_b.add_trace(go.Line(x=lower.index, y=lower, name="-2σ", line=dict(dash="dot")))
    fig_b.update_layout(template="plotly_white", title=f"Bollinger (retornos) - {crypto_single}")

    # Animation: cumulative price over time (simple animation)
    # build long df from P_weeklyC
    df_long = P_weeklyC.melt(id_vars="Date", var_name="Crypto", value_name="Price").dropna()
    df_long = df_long.sort_values(["Crypto","Date"])
    # For animation performance: filter selected cryptos
    df_anim = df_long[df_long["Crypto"].isin(cryptos_sel)]
    fig_anim = px.line(df_anim, x="Date", y="Price", color="Crypto", animation_frame="Date",
                       title="Evolución animada de precios (frame = fecha)")
    fig_anim.update_layout(template="plotly_white")

    # metrics for selected cryptos
    metrics = []
    for c in (cryptos_sel or []):
        s = R_weeklyC[c].dropna()
        metrics.append({
            "Cripto": c,
            "CVaR95": round(cvar_historical(s,0.95),6),
            "MaxDrawdown": round(max_drawdown(s),6),
            "SharpeAnn": round(sharpe_ann(s),6)
        })
    cols = [{"name":k,"id":k} for k in metrics[0].keys()] if metrics else []
    return fig_b, fig_anim, metrics, cols

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
