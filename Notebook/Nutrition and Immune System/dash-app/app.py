# import plotly.express as px
import dash
import plotly.graph_objects as go
# , State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import helper as hp
import os
from dash.dependencies import Input, Output
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Constants
countries = hp.Fat_Supply_Quantity_Data['Country'].to_list()

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


@app.callback(
    Output('bar_chart_of_food_1', 'figure'),
    [Input('Radio_foodType_tab2', 'value'),
     Input('dropdown_country', 'value'),
     Input('number_of_output', 'value')
     ])
def plot_barcahr_of_food(bar_chart_of_food_1, dropdown_country, number_of_output):
    from utils import NutritionData
    # print(os.path.join('data', bar_chart_of_food_1+'.csv'))
    df = NutritionData(os.path.join('data', bar_chart_of_food_1 + '.csv')).data_frame
    # print(df.head())
    df_country = df.loc[dropdown_country]

    # print(df_country)

    scalar_model = os.path.join('data/models',
                                os.path.join(number_of_output,
                                             bar_chart_of_food_1 + '_scalar.pkl'))
    # print(scalar_model)
    main_model = os.path.join('data/models',
                              os.path.join(number_of_output,
                                           bar_chart_of_food_1 + '.pkl'))
    # print(main_model)
    if os.path.exists(scalar_model) and os.path.exists(main_model):
        from utils import predict
        from utils import top8output
        from utils import top23output
        output = predict(scalar_model, main_model, [[0, 0, 0, 1, 0.5, 0.5]])[0]
        if number_of_output == '8output':
            output_label = top8output

        else:
            output_label = top23output

        df_country_filtered = df_country[output_label]
        df_country_filtered = df_country_filtered.reset_index(drop=True).to_list()
        fig = go.Figure([
            go.Bar(name='Predicted (For high immunity)', x=output_label, y=output),
            go.Bar(name=dropdown_country, x=output_label, y=df_country_filtered)
        ])

        fig.update_layout(
            barmode='group',
            title="Food habit of total population ",
            yaxis_title="Population(%)",
            xaxis_title="Food item",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="RebeccaPurple"
            )
        )

        return fig
        # print(top8output)


if __name__ == '__main__':
    app.run_server(debug=True)
