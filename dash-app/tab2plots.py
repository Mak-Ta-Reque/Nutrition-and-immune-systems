import plotly.graph_objects as go
import os


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
