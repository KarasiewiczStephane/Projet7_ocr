# create me a minimal dash application 
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import requests
from dash import Input,Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


fake_df = requests.get("http://localhost:8081/df").json()
ID_client = eval(requests.get("http://localhost:8081/get_ID_client").text)

df=pd.DataFrame(fake_df)
df = df.set_index("SK_ID_CURR")

# create a dash application
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

row = html.Div([
    dbc.Row([
        html.H1("Dashboard solvabilité",style={"text-align":"center"}),
        ]),
    dbc.Row([
        dbc.Col(
            html.H2("Sélection ID Clients"), width={"size": 2, "offset": 1}),
        dbc.Col(
            html.Div(
                dcc.Dropdown(ID_client,ID_client[0],id='ID_client_dropdown')), width=8)
            ]),
    dbc.Row([
        dbc.Col(
            html.H3("Information Clients", style={"color":"blue"}), width={"size": 6}),
        dbc.Col(
            html.H3("Statut du prêt"), width=6)
            ]),
    dbc.Row([
        dbc.Col(
            dbc.ListGroup(
            [
                dbc.ListGroupItem("Age", id="age"),
                dbc.ListGroupItem("Statut Familial", id="statut_familial"),
                dbc.ListGroupItem("Genre", id="genre"),
                dbc.ListGroupItem("Revenu", id="revenu")
            ]
                ), width=6),
        dbc.Col([
            dcc.Graph(id= "jauge")
                            ])
            ]),
    dbc.Row([
        dbc.Col(
            html.Div("Détails de la décision"), width=12)
            ]),
    dbc.Row([
        dbc.Col(
            html.Div("Graph shap"), width=12)
            ]),
    dbc.Row([
        dbc.Col(
            html.Div("distribtution variable"), width=2),
        dbc.Col(
            html.Div(
                dcc.Dropdown(ID_client,ID_client[0],id='ID_client_dropdown2')), width=2)
            ])
        ])


def generate_table(df, ID_client_dropdown):
    client = df.loc[ID_client_dropdown]
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in client.index[:4]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(client[col]) for col in client.index[:4]
                                      ])
        ])
    ])




@app.callback(
    Output('jauge', 'figure'),
    Input('ID_client_dropdown', 'value'))
def print_client(ID_client_dropdown):
    solvabilite = float(requests.get(f"http://localhost:8081/predict_solvabilite/{ID_client_dropdown}").text)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        gauge = {"axis" : {"range" : [0,1]}},
        value = solvabilite,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Solvabilite Client"}))
    return fig 



@app.callback(
    Output('genre', 'children'),
    Output('age', 'children'),
    Output('statut_familial', 'children'),
    Output('revenu', 'children'),
    Input('ID_client_dropdown', 'value'))
def print_info_client(ID_client_dropdown):
    info_client = requests.get(f"http://localhost:8081/get_info_client/{ID_client_dropdown}").json()
    info_client = pd.Series(info_client)

    if info_client["CODE_GENDER"]== 1:
        message_genre = 'sexe: Homme'
    else:
        message_genre = 'sexe: Femme'
        
    return message_genre,message_genre,message_genre,message_genre


# create a dash html div for the scatter plot
app.layout = row

# run the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
    #app.run_server(debug=True, port=80)

