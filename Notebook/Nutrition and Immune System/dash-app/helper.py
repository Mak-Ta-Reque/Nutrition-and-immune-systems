import os
import pandas as pd
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate
import plotly.express as px
import pathlib
import plotly.graph_objs as go
import numpy as np

# Replce any nonnumaric char from a column.

Fat_Supply_Quantity_Data=pd.read_csv(r"data/Fat_Supply_Quantity_Data.csv",encoding='utf8')
Food_Supply_kcal_Data=pd.read_csv(r"data/Food_Supply_kcal_Data.csv",encoding='utf8')
Food_Supply_Quantity_kg_Data=pd.read_csv(r"data/Food_Supply_Quantity_kg_Data.csv",encoding='utf8')
Protein_Supply_Quantity_Data=pd.read_csv(r"data/Protein_Supply_Quantity_Data.csv",encoding='utf8')

Fat_Supply_Quantity_Data = Fat_Supply_Quantity_Data.replace(regex=r'^<', value=2.5)
Food_Supply_kcal_Data = Food_Supply_kcal_Data.replace(regex=r'^<', value=2.5)
Food_Supply_Quantity_kg_Data = Food_Supply_Quantity_kg_Data.replace(regex=r'^<', value=2.5)
Protein_Supply_Quantity_Data = Protein_Supply_Quantity_Data.replace(regex=r'^<', value=2.5)


column_names=Fat_Supply_Quantity_Data.columns

df=None

def scatter_plot(df,Radio_differentFoods,Radio_Output):
    fig=px.scatter(df, x=Radio_differentFoods, y=Radio_Output, trendline="ols",title ="X Axis")
    return fig
def box_plot(df):
    fig=px.box(df)
    return fig

def scatter_plot_tab_1(Radio_foodType,Radio_differentFoods,Radio_Output):
    figure=None
    if Radio_foodType=="Protine":
        df = Protein_Supply_Quantity_Data
        figure=scatter_plot(df, Radio_differentFoods,Radio_Output)

    elif Radio_foodType=="Fat":
        df = Fat_Supply_Quantity_Data
        figure=scatter_plot(df, Radio_differentFoods,Radio_Output)

    elif Radio_foodType=="KCal":
        df = Food_Supply_kcal_Data
        figure=scatter_plot(df, Radio_differentFoods,Radio_Output)

    elif Radio_foodType=="Quantity":
        df = Food_Supply_Quantity_kg_Data
        figure=scatter_plot(df, Radio_differentFoods,Radio_Output)
    else:
        pass
    return figure

def box_plot_tab_1(Radio_foodType):
    figure=None
    if Radio_foodType=="Protine":
        df = Protein_Supply_Quantity_Data
        figure=box_plot(df.iloc[:,1:24])

    elif Radio_foodType=="Fat":
        df = Fat_Supply_Quantity_Data
        figure=box_plot(df.iloc[:,1:24])

    elif Radio_foodType=="KCal":
        df = Food_Supply_kcal_Data
        figure=box_plot(df.iloc[:,1:24])

    elif Radio_foodType=="Quantity":
        df = Food_Supply_Quantity_kg_Data
        figure=box_plot(df.iloc[:,1:24])
    else:
        pass
    return figure






