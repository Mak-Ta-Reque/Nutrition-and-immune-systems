import pandas as pd
import plotly.express as px

# Replce any nonnumaric char from a column.

Fat_Supply_Quantity_Data = pd.read_csv(r"data/Fat_Supply_Quantity_Data.csv", encoding='utf8')
Food_Supply_kcal_Data = pd.read_csv(r"data/Food_Supply_kcal_Data.csv", encoding='utf8')
Food_Supply_Quantity_kg_Data = pd.read_csv(r"data/Food_Supply_Quantity_kg_Data.csv", encoding='utf8')
Protein_Supply_Quantity_Data = pd.read_csv(r"data/Protein_Supply_Quantity_Data.csv", encoding='utf8')


def numeric_converter(dataframe):
    converted = dataframe.replace(regex=r'^<', value=2.5)
    converted["Undernourished"] = pd.to_numeric(converted["Undernourished"], downcast="float")
    return converted


Fat_Supply_Quantity_Data = numeric_converter(Fat_Supply_Quantity_Data)
Food_Supply_kcal_Data = numeric_converter(Food_Supply_kcal_Data)
Food_Supply_Quantity_kg_Data = numeric_converter(Food_Supply_Quantity_kg_Data)
Protein_Supply_Quantity_Data = numeric_converter(Protein_Supply_Quantity_Data)
column_names = Fat_Supply_Quantity_Data.columns

# Fat_Supply_Quantity_Data=Fat_Supply_Quantity_Data.reset_index(drop=True, inplace=True)
# group=Fat_Supply_Quantity_Data.groupby("Country")
# x=group.get_group("Bangladesh")
# x.reset_index(drop=True, inplace=True)
# group=Fat_Supply_Quantity_Data
# group.reset_index(drop=True, inplace=True)
# getGroup=group.get_group("Bangladesh").T
# x=x.T
# x=x[2:24]
# z=[]
# y=[]
#
# for each in x.index[1:24]:
#     # print(each)
#     y.append(each)
# for each in x.iloc[1:24,0]:
#     z.append(each)
#
# d = {'Name':y,'Values':z}
# dfff=pd.DataFrame(d)
# print(dfff)
# print(column_names)
# print(column_names[1:24])


"""Tab 1 Starts"""


def scatter_plot(df, Radio_differentFoods, Radio_Output):
    fig = px.scatter(df, x=Radio_differentFoods, y=Radio_Output, trendline="ols", title="X Axis")
    return fig


def box_plot(df):
    fig = px.box(df)
    return fig


def scatter_plot_tab_1(Radio_foodType, Radio_differentFoods, Radio_Output):
    figure = None
    correlation = None
    if Radio_foodType == "Protine":
        df = Protein_Supply_Quantity_Data
        figure = scatter_plot(df, Radio_differentFoods, Radio_Output)
        correlation = df[Radio_differentFoods].corr(df[Radio_Output])
    elif Radio_foodType == "Fat":
        df = Fat_Supply_Quantity_Data
        figure = scatter_plot(df, Radio_differentFoods, Radio_Output)
        correlation = df[Radio_differentFoods].corr(df[Radio_Output])
    elif Radio_foodType == "KCal":
        df = Food_Supply_kcal_Data
        figure = scatter_plot(df, Radio_differentFoods, Radio_Output)
        correlation = df[Radio_differentFoods].corr(df[Radio_Output])
    elif Radio_foodType == "Quantity":
        df = Food_Supply_Quantity_kg_Data
        figure = scatter_plot(df, Radio_differentFoods, Radio_Output)
        correlation = df[Radio_differentFoods].corr(df[Radio_Output])
    else:
        pass

    figure.update_layout(title={
        'text': "Correlation between " + str(Radio_differentFoods) + " and " + str(Radio_Output) + " is " + str(
            correlation),
        'y': 0.01,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'bottom'})
    return figure


def box_plot_tab_1(Radio_foodType):
    figure = None
    if Radio_foodType == "Protine":
        df = Protein_Supply_Quantity_Data
        figure = box_plot(df.iloc[:, 1:24])

    elif Radio_foodType == "Fat":
        df = Fat_Supply_Quantity_Data
        figure = box_plot(df.iloc[:, 1:24])

    elif Radio_foodType == "KCal":
        df = Food_Supply_kcal_Data
        figure = box_plot(df.iloc[:, 1:24])

    elif Radio_foodType == "Quantity":
        df = Food_Supply_Quantity_kg_Data
        figure = box_plot(df.iloc[:, 1:24])
    else:
        pass
    return figure


""" Tab 1 Finished"""


def pie_chart(dropDown,dff):
    group = dff.groupby("Country")
    getGroup = group.get_group(dropDown)
    x = getGroup.T
    y = []
    z = []
    for each in x.index[1:24]:
        y.append(each)
    for each in x.iloc[1:24, 0]:
        z.append(each)
    d = {'Name': y, 'Values': z}
    df = pd.DataFrame(d)
    figure = px.pie(df, values='Values', names='Name')
    return figure


def pie_chart_1_for_app(dropDown, Radio_foodType):
    figure = None
    if Radio_foodType == "Protine":
        dff = Protein_Supply_Quantity_Data
        figure = pie_chart(dropDown,dff)
    elif Radio_foodType == "Fat":
        dff = Fat_Supply_Quantity_Data
        figure = pie_chart(dropDown,dff)
    elif Radio_foodType == "KCal":
        dff = Food_Supply_kcal_Data
        figure = pie_chart(dropDown,dff)
    elif Radio_foodType == "Quantity":
        dff = Food_Supply_Quantity_kg_Data
        figure = pie_chart(dropDown,dff)
    else:
        pass
    return figure
