# import plotly.express as px
import dash
# , State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import helper as hp
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# Constants
countries = hp.Fat_Supply_Quantity_Data['Country'].to_list()

app.title = "Nutrition and Immune System"

app.layout = html.Div(children=[
    html.Div([
        html.P("Nutration and Immune System",
               style={
                   'display': 'center',
                   # 'margin-left': '100px',
                   # 'margin-top': '50px',
                   'text-align': 'center',
                   'fontSize': 40
               },
               ),
        html.P(["The purpose of this software is to compare different food products"
                " consumption in ideal situation (requirement for boosting immunity)"
                "and current situation of different countries in this time of COVID-19 pandemic."
                   , html.Br(),
                " In 'Data Visualization' page, we can find out the correlation between different food item"
                " with the correlation of different health condition. Also user can compare"
                " between two countries about their percentage of different type of food having and"
                " health condition depending on their food habit.", html.Br(),
                "In the 'Prediction' page is the prediction of different countries"
                " current food consumption and what type of food should they add more"
                " or decrease from their eating menu depending upon their different health"
                " condition."],
               style={
                   'display': 'inline-block',
                   'margin-left': '20px',
                   # 'margin-top': '50px',
                   'text-align': 'left',
                   'fontSize': 20,
                   'margin-bottom': '20px'
               }
               )
    ]),
    dcc.Tabs([
        dcc.Tab(label='Data Visualization', children=[
            html.Div([
                html.Div([
                    html.P("Choose the type of Nutrition Content and Calorie Consumption you want to check:",
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
                html.P("Choose different type foods items:",
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
                html.P("General Health Condition:",
                       style={'display': 'inline-block',
                              'margin-left': '35px',
                              'margin-top': '50px',
                              'fontSize': 20})
            ]),
            html.Div([dcc.RadioItems(
                id="Radio_Output",
                options=[
                    # {'label': hp.column_names[i], 'value': hp.column_names[i]}
                    # for i in range(24, 30)

                    {'label': "Obesity", "value": "Obesity"},
                    {'label': "Undernourished", "value": "Undernourished"},
                    {'label': "Covid-19 Confirmed Cases", "value": "Confirmed"},
                    {'label': "Covid-19 Death Cases", "value": "Deaths"},
                    {'label': "Recovered from Covid-19", "value": "Recovered"},
                    {'label': "Covid-19 Active Cases", "value": "Active"}

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
                )]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="Tab_1_graph_5"
                    )], className="pretty_container six columns",
                ),
                html.Div([
                    dcc.Graph(
                        id="Tab_1_graph_6"
                    )], className="pretty_container six columns",
                )])
        ]),

        dcc.Tab(label='Prediction', children=[

            html.Div([
                html.Div([
                    html.P("Choose the type of Nutrition Content and Calorie Consumption you want to check:",
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
                            'margin-top': '10px',
                            'margin-bottom': '10px',
                            # 'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),

            html.Div([
                html.P("Please select a country for which you want to produce food:", className="control_label_2",
                    style={'display': 'inline-block',
                            'margin-left': '35px',
                            'margin-top': '10px',
                            'margin-bottom': '10px',
                            # 'margin-bottom': '-100px',
                            'fontSize': 20}),
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
                    {'label': 'All type of foods', 'value': '23output'},
                    {'label': 'Top 8 foods', 'value': '8output'},
                ],
                value='8output',
                labelStyle={'display': 'inline-block',
                            'margin-left': '30px',
                            'margin-top': '20px',
                            'margin-bottom': '-100px',
                            'fontSize': 15}
            )]),

            html.Div([
                dcc.Graph(
                    id="bar_chart_of_food_1"
                ),
            ]),
            html.Div([
                html.P(["In the upper plot, blue bar represents the perfect amount of"
                        " food the selected country should eat to have a good immune system"
                        " and the red bar represents the amount of food they are eating now."
                        " Here the goal is that the red bar is in equal height with the blue bar for different foods to get better immune system."
                        " So, from here, we can easily decide which type of food should we add"
                        " more to menu and cut from the menu for having a better immune system in"
                        " this challenging time of COVID-19 pandemic."],
                    style={
                   'display': 'inline-block',
                   'margin-top': '30px',
                   'margin-left': '20px',
                   'text-align': 'left',
                   'fontSize': 20,
               })
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
    data = hp.pie_chart_1_and_2_for_app(dropDown, radio)
    return data


@app.callback(
    Output('Tab_1_graph_4', 'figure'),
    [Input('radio_pie', 'value'),
     Input('Tab_1_dropdown_2', 'value')])
def pie_chart_2(radio, dropDown):
    data = hp.pie_chart_1_and_2_for_app(dropDown, radio)
    return data


@app.callback(
    Output('Tab_1_graph_5', 'figure'),
    [Input('radio_pie', 'value'),
     Input('Tab_1_dropdown_1', 'value')])
def pie_chart_3(radio, dropDown):
    data = hp.pie_chart_3_and_4_for_app(dropDown, radio)
    return data


@app.callback(
    Output('Tab_1_graph_6', 'figure'),
    [Input('radio_pie', 'value'),
     Input('Tab_1_dropdown_2', 'value')])
def pie_chart_4(radio, dropDown):
    data = hp.pie_chart_3_and_4_for_app(dropDown, radio)
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
