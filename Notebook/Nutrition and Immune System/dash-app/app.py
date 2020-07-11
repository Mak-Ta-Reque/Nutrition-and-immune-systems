# import plotly.express as px
import dash
# , State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import helper as hp
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    dcc.Tabs([
        dcc.Tab(label='Tab one', children=[
            html.Div([
                html.Div([
                    html.P("Food Type:",
                           style={'display': 'inline-block',
                                  'margin-left': '35px',
                                  'margin-top': '50px',
                                  'fontSize': 20})])
            ]),
            html.Div([dcc.RadioItems(
                id="Radio_foodType",
                options=[
                    {'label': 'Protein', 'value': 'Protine'},
                    {'label': 'Fat', 'value': 'Fat'},
                    {'label': 'KCal', 'value': 'KCal'},
                    {'label': 'Quantity', 'value': 'Quantity'}
                ],
                value='Protine',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),
            html.Div([
                html.P("Different Foods:",
                       style={'display': 'inline-block',
                              'margin-left': '35px',
                              'margin-top': '50px',
                              'fontSize': 20})
            ]),
            html.Div([dcc.RadioItems(
                id="Radio_differentFoods",
                options=[
                    {'label': hp.column_names[i], 'value': hp.column_names[i]}
                    for i in range(1, 24)
                ],
                value='Alcoholic Beverages',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),
            html.Div([
                html.P("Output:",
                       style={'display': 'inline-block',
                              'margin-left': '35px',
                              'margin-top': '50px',
                              'fontSize': 20})
            ]),
            html.Div([dcc.RadioItems(
                id="Radio_Output",
                options=[
                    {'label': hp.column_names[i], 'value': hp.column_names[i]}
                    for i in range(24, 30)
                ],
                value='Obesity',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),
            html.Div([
                dcc.Graph(
                    id="graph_tab_1"
                ),
            ]),
            html.Div([
                dcc.Graph(
                    id="graph_tab_1_2"
                ),
            ])

        ]),
        dcc.Tab(label='Tab two', children=[
            dcc.Graph(
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [1, 4, 1],
                         'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [1, 2, 3],
                         'type': 'bar', 'name': u'Montréal'},
                    ]
                }
            )
        ]),
        dcc.Tab(label='Tab three', children=[
            dcc.Graph(
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [2, 4, 3],
                         'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [5, 4, 3],
                         'type': 'bar', 'name': u'Montréal'},
                    ]
                }
            )
        ]),
    ])
])


# def update_figure Bar
@app.callback(
    Output('graph_tab_1', 'figure'),
    [Input('Radio_foodType', 'value'),
     Input('Radio_differentFoods', 'value'),
     Input('Radio_Output', 'value')])
def update_figure_scatter_plot(Radio_foodType, Radio_differentFoods, Radio_Output):
    data = hp.scatter_plot_tab_1(Radio_foodType, Radio_differentFoods, Radio_Output)
    return data

# def update_figure Bar
@app.callback(
    Output('graph_tab_1_2', 'figure'),
    [Input('Radio_foodType', 'value')])
def update_figure_box_plot(Radio_foodType):
    data = hp.box_plot_tab_1(Radio_foodType)
    return data


if __name__ == '__main__':
    app.run_server(debug=True)
