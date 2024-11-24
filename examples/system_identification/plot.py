from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Simple stock plot with fixed axis'),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"),
    Input("graph", "id"))
def display_graph(_):
    df = pd.read_csv('actor_loss.csv')  # replace with your own data source
    fig = px.line(df, x='Step', y='Loss')
    return fig

app.run_server(debug=True)
