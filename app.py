from dash import Dash, dcc, html, Input, Output

import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

from dash_bootstrap_templates import load_figure_template
load_figure_template("materia")

import pandas as pd
import numpy as np

import json

# ----------------------------------------------------------------------

datadir = "data"
datadate = '2022-05-24'

df = pd.read_csv('{}/klusterit-{}.csv'.format(datadir, datadate))
dfvars = pd.read_csv('{}/klusterimuuttujat2-{}.csv'.format(datadir,
                                                           datadate),
                     sep=";")
dfcc = pd.read_csv('{}/kmeans_centers-{}.csv'.format(datadir, datadate),
                   sep=";", index_col=0)
dfv = pd.read_csv('{}/kmeans_muuttujat-{}.csv'.format(datadir,
                                                      datadate),
                  sep=";", index_col=0)
dfbg = pd.read_csv('{}/kmeans_taustamuuttujat-{}.csv'.format(datadir,
                                                             datadate),
                   sep=";", index_col=0)
dffac = pd.read_csv('{}/faktorit-{}.csv'.format(datadir, datadate))

with open('{}/descriptions-{}.json'.format(datadir, datadate)) as json_file:
    descriptions = json.load(json_file)
df['description']=descriptions['clusters']
df['longdescription']=descriptions['cluster_descriptions']
with open('{}/dimensions-{}.json'.format(datadir, datadate)) as json_file:
    dimensions = json.load(json_file)
with open('{}/variables-{}.json'.format(datadir, datadate)) as json_file:
    variables = json.load(json_file)

pos_color = "#2196f3"
neg_color = "#e51c23"
bgr_color = "paleturquoise"
    
# ----------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

server = app.server

app.layout = dbc.Container([
    dbc.NavbarSimple(brand='Klusterikortti, aineisto: "Marketing Campaign"',
                     brand_href="#", color="primary", dark=True,),
    html.Hr(),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='selected_cluster',
        options=[{'label': 'Klusteri {}: "{}"'.format(int(r['klusteri']),
                                                      r['description']),
                  'value': int(r['klusteri'])} for _, r in df.iterrows()],
        value=df.klusteri[0]), width=10),
             dbc.Col(html.Div(id='fraction', style={'font-size': 'large'}),
                     width=2),
             ], align="center",),
    dbc.Row([dbc.Col(html.Div(id='longdescription',
                              style={'font-size': 'x-large', "padding": "1rem 1rem",
                                     "background-color": bgr_color}),
                     width={"size": 8, "offset": 1},),
             dbc.Col([html.Div(id='income',
                               style={'font-size': 'large', "padding": "0.5rem 0.5rem"}),
                      html.Div(id='age',
                               style={'font-size': 'large', "padding": "0.5rem 0.5rem"}),
                      ],
                      width={"size": 2, "offset": 1},)
             ],
            style={"padding": "1rem 1rem",}),
    dbc.Row([dbc.Col(dcc.Graph(id="stiglitz"), width=12), ### 9
             ### dbc.Col(dcc.Graph(id="sex"), width=3),
             ]),
    dbc.Row([dbc.Col(dcc.Graph(id="smallest_vars"), width=3),
             dbc.Col(dcc.Graph(id="largest_vars"), width=3),
             dbc.Col(dcc.Graph(id="factors"), width=6),
             ]),
    html.Hr(),
    ], fluid=True)

# ----------------------------------------------------------------------

@app.callback(
    Output("stiglitz", "figure"), 
    Input("selected_cluster", "value"))
def update_bar_chart(cl):
    df2 = df[df.klusteri==cl].transpose()
    df2 = df2[1:9]
    df2.columns = ['data']
    df2["color"] = np.where(df2['data']<0, neg_color, pos_color)
    df2['description'] = df2.index.to_series().apply(lambda x: dimensions[x]['description'])
    fig = go.Figure(data=go.Bar(x=df2.index, y=df2.data,
                                marker_color=df2.color,
                                customdata=df2.description,
                                hovertemplate='%{y:.2f}: %{customdata}<extra></extra>'))
    fig.update_yaxes(range=[-1.05, 1.05])
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_layout(title_text='Stiglitz-ulottuvuudet')
    return fig

@app.callback(
    Output("fraction", "children"), 
    Input("selected_cluster", "value"))
def update_fraction(cl):
    df2 = df[df.klusteri==cl]
    return f'Osuus: {round(df2.osuus.values[0]*100)} %'

@app.callback(
    Output("longdescription", "children"), 
    Input("selected_cluster", "value"))
def update_longdescription(cl):
    df2 = df[df.klusteri==cl]
    return df2['longdescription']

@app.callback(
    Output("income", "children"), 
    Input("selected_cluster", "value"))
def update_fraction(cl):
    return f'Keskim. tulot: ${round(dfv.Income[cl-1])}'

@app.callback(
    Output("age", "children"), 
    Input("selected_cluster", "value"))
def update_fraction(cl):
    return f'Keskim. ikä: {round(dfv.Age[cl-1]-10)}'

#@app.callback(
#    Output("sex", "figure"), 
#    Input("selected_cluster", "value"))
#def update_sex_chart(cl):
#    sp = dfbg.sukupuoli[cl-1]
#    fig = go.Figure(data=[go.Pie(labels=["poika", "tyttö"], values=[2-sp, sp-1], sort=False)])
#    fig.update_layout(title_text="Sukupuoli")
#    return fig

@app.callback(
    Output("largest_vars", "figure"), 
    Input("selected_cluster", "value"))
def update_largest_vars(cl):
    largest = dfcc.loc[cl].sort_values(ascending=False)[:10].sort_values()
    colorvec = np.where(largest<0, neg_color, pos_color)
    description = largest.index.to_series().apply(lambda x: variables[x]['description'])
    fig = go.Figure(data=go.Bar(y=[s[:20] for s in largest.index],
                                x=largest.values, orientation='h',
                                marker_color=colorvec, customdata=description,
                                hovertemplate='%{x:.2f}: %{customdata}<extra></extra>'))
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_layout(title_text="Suurimmat muuttujat")
    return fig

@app.callback(
    Output("smallest_vars", "figure"), 
    Input("selected_cluster", "value"))
def update_smallest_vars(cl):
    smallest = dfcc.loc[cl].sort_values(ascending=False)[-10:]
    colorvec = np.where(smallest<0, neg_color, pos_color)
    description = smallest.index.to_series().apply(lambda x: variables[x]['description'])
    fig = go.Figure(data=go.Bar(y=[s[:20] for s in smallest.index],
                                x=smallest.values, orientation='h',
                                marker_color=colorvec, customdata=description,
                                hovertemplate='%{x:.2f}: %{customdata}<extra></extra>'))
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_layout(title_text="Pienimmät muuttujat")
    return fig

@app.callback(
    Output("factors", "figure"), 
    Input("selected_cluster", "value"))
def update_factor_scatter(cl):
    dffac["klusteristr"] = dffac["klusteri"].astype(str)
    all = list(range(1,99))
    all.remove(cl)
    dffac["klusteristr"] = dffac["klusteristr"].replace([str(x) for x in all],
                                                        '99')
    fig = px.scatter(dffac, x="faktori1", y="faktori2",
                     color='klusteristr',
                     color_discrete_map = {str(cl): pos_color,
                                           '99': bgr_color},
                     title="Faktorivisualisointi",
                     labels={"faktori1": descriptions['factors'][0],
                             "faktori2": descriptions['factors'][1]})
    fig.update_layout(showlegend=False)
    return fig

# ----------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)

# ----------------------------------------------------------------------
