# import plotly.express as px
import dash
# , State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import helper as hp
from dash.dependencies import Input, Output
import flask




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# Constants
countries = hp.Fat_Supply_Quantity_Data['Country'].to_list()

app.layout = html.Div(children=[
    dcc.Tabs([
        dcc.Tab(label='Data visualization', children=[
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
            ]),
            html.Div([
                html.P("Food Type:",
                       style={'display': 'inline-block',
                              'margin-left': '35px',
                              'margin-top': '50px',
                              'fontSize': 20})
            ]),
            html.Div(
                [
                    dcc.RadioItems(
                        id="radio_pie",
                        options=[
                            {'label': 'Protein', 'value': 'Protine'},
                            {'label': 'Fat', 'value': 'Fat'},
                            {'label': 'KCal', 'value': 'KCal'},
                            {'label': 'Quantity', 'value': 'Quantity'}
                        ],
                        value='Protine',
                        labelStyle={'display': 'inline-block',
                                    'margin-left': '5px',
                                    'fontSize': 15}
                    )], className="pretty_container six columns",
            ),
            html.Div([
                html.Div(
                    [
                        html.P("Select 1st Country:", className="control_label_1"),
                        dcc.Dropdown(
                            id="Tab_1_dropdown_1",
                            options=[
                                {"label": country, "value": country}
                                for country in hp.Fat_Supply_Quantity_Data['Country']
                            ],
                            value='Germany',
                            clearable=False,
                            multi=False
                        )

                    ],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [
                        html.P("Select 2nd Country:", className="control_label_2"),
                        dcc.Dropdown(
                            id="Tab_1_dropdown_2",
                            options=[
                                {"label": country, "value": country}
                                for country in hp.Fat_Supply_Quantity_Data['Country']
                            ],
                            value='Germany',
                            clearable=False,
                            multi=False
                        )

                    ],
                    className="pretty_container six columns",
                ),
            ],
                style={
                    "margin-top": "35px",
                    "margin-left": "10px"
                }),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="Tab_1_graph_3"
                    )], className="pretty_container six columns",
                ),
                html.Div([
                    dcc.Graph(
                        id="Tab_1_graph_4"
                    )], className="pretty_container six columns",
                )])
        ]),

        dcc.Tab(label='Prediction', children=[

            html.Div([
                html.Div([
                    html.P("Choose the nutrition/content:",
                           style={'display': 'inline-block',
                                  'margin-left': '35px',
                                  'margin-top': '50px',
                                  'fontSize': 20})])
            ]),

            html.Div([dcc.RadioItems(
                id="Radio_foodType_tab2",
                options=[
                    {'label': 'Protein', 'value': 'Protein_Supply_Quantity_Data'},
                    {'label': 'Fat', 'value': 'Fat_Supply_Quantity_Data'},
                    {'label': 'KCal', 'value': 'Food_Supply_kcal_Data'},
                    {'label': 'Quantity', 'value': 'Food_Supply_Quantity_kg_Data'}
                ],
                value='Protein_Supply_Quantity_Data',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),


            html.Div([
                html.P("Please select a country for which you want to produce food", className="control_label_2"),
                dcc.Dropdown(
                    id='dropdown_country',
                    options=[
                        {'label': i, 'value': i}
                        for i in countries
                    ],
                    clearable=False,
                    value='Germany'
                )]),
            html.Div([dcc.RadioItems(
                id="number_of_output",
                options=[
                    {'label': 'All output', 'value': '23output'},
                    {'label': 'Top 8', 'value': '8output'},
                ],
                value='8output',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),

            html.Div([
                dcc.Graph(
                    id="bar_chart_of_food_1"
                ),
            ])
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


@app.callback(
    Output('Tab_1_graph_3', 'figure'),
    [Input('radio_pie', 'value'),
     Input('Tab_1_dropdown_1', 'value')])
def pie_chart_1(radio, dropDown):
    data = hp.pie_chart_1_for_app(dropDown, radio)
    return data


@app.callback(
    Output('Tab_1_graph_4', 'figure'),
    [Input('radio_pie', 'value'),
     Input('Tab_1_dropdown_2', 'value')])
def pie_chart_2(radio, dropDown):
    data = hp.pie_chart_1_for_app(dropDown, radio)
    return data


@app.callback(
    Output('bar_chart_of_food_1', 'figure'),
    [Input('Radio_foodType_tab2', 'value'),
     Input('dropdown_country', 'value'),
     Input('number_of_output', 'value')
     ])
def plot_barcahr_of_food(bar_chart_of_food_1, dropdown_country, number_of_output):
    from tab2plots import plot_barcahr_of_food as bar_plot
    return bar_plot(bar_chart_of_food_1, dropdown_country, number_of_output)




if __name__ == '__main__':
    app.run_server(debug=True)
