# import pip
#
# pip.main(['install', 'dash', 'pandas', 'plotly', 'numpy'])

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import os

current_directory = os.getcwd()
data_dir = current_directory + '/data/'

app = Dash(__name__)

tab1 = html.Div([
    html.H3('PCA'),
    dcc.Dropdown(id='pca-input',
                 options=[{'label': 'Two Dimensional PCA', 'value': 2},
                          {'label': 'Three Dimensional PCA', 'value': 3}],
                 value=2),
    html.Img(id='pca-output', style={'height': '30%', 'width': '40%'})
])

tab2 = html.Div([
    html.H3('Validation Curve'),
    dcc.Dropdown(id='validation-curve-input',
                 options=[{'label': 'K Nearest Neighbors', 'value': 'k_nearest_neighbors'},
                          {'label': 'Decision Tree Classifier', 'value': 'decision_tree_classifier'},
                          {'label': 'Random Forest Classifier', 'value': 'random_forest_classifier'}],
                 value='k_nearest_neighbors'),
    html.Img(id='validation-curve-output', style={'height': '30%', 'width': '40%'})
])

tab3 = html.Div([
    html.H3('Learning Curve'),
    dcc.Dropdown(id='learning-curve-input',
                 options=[{'label': 'K Nearest Neighbors', 'value': 'k_nearest_neighbors'},
                          {'label': 'Logistic Regression', 'value': 'logistic_regression'},
                          {'label': 'Support Vector Classifier', 'value': 'support_vector_classifier'},
                          {'label': 'Decision Tree Classifier', 'value': 'decision_tree_classifier'},
                          {'label': 'Random Forest Classifier', 'value': 'random_forest_classifier'},
                          {'label': 'Gradient Boosting Classifier', 'value': 'gradient_boosting_classifier'}],
                 value='k_nearest_neighbors'),
    html.Img(id='learning-curve-output', style={'height': '30%', 'width': '40%'})
])

tab4 = html.Div([
    html.H3('Elbow Plot'),
    html.Img(src='assets/elbow_plot.png')
])

tab5 = html.Div([
    html.H3('Silhouette Plot'),
    dcc.Dropdown(id='silhouette-plot-input',
                 options=[{'label': 'k = 2', 'value': '2'},
                          {'label': 'k = 3', 'value': '3'},
                          {'label': 'k = 4', 'value': '4'},
                          {'label': 'k = 5', 'value': '5'}],
                 value='2'),
    html.Img(id='silhouette-plot-output', style={'height': '30%', 'width': '40%'})
])


def render_imbalance_plot():
    data = pd.read_csv(data_dir + 'application_train.csv',
                       usecols=['SK_ID_CURR', 'TARGET'])
    fig = px.histogram(data, x='TARGET')
    return fig


tab6 = html.Div([
    html.H3('Class Imbalance Plot'),
    dcc.Graph(figure=render_imbalance_plot())
])

# tab7 = html.Div([
#     html.H3('Residuals Plot'),
#     dcc.Dropdown(id='select-model-dropdown-input',
#                  options=[{'label': 'One', 'value': 'One'},
#                           {'label': 'Two', 'value': 'Two'}]),
#     html.Div(id='select-model-dropdown-output')
# ])
#
# tab8 = html.Div([
#     html.H3('Prediction Error Plot'),
#     dcc.Dropdown(id='select-model-dropdown-input',
#                  options=[{'label': 'One', 'value': 'One'},
#                           {'label': 'Two', 'value': 'Two'}]),
#     html.Div(id='select-model-dropdown-output')
# ])


def find_numeric():
    data = pd.read_csv(data_dir + 'application_train.csv')
    cols = data.columns.to_list()
    remove_cols = ['SK_ID_CURR', 'TARGET']

    num_cols = []
    for col in cols:
        if col not in remove_cols:
            if isinstance(data[col][0], np.floating) or isinstance(data[col][0], np.int64):
                num_cols.append(col)
    return num_cols


outlier_columns = find_numeric()
tab9 = html.Div([
    html.H3('Box Plot'),
    dcc.Dropdown(outlier_columns, outlier_columns[0], id='outlier-input'),
    dcc.Graph(id='outlier-output')
])


tab10 = html.Div([
    html.H3('Feature Importance Plot (Random Forest Classifier)'),
    dcc.Dropdown(id='feature-importance-input',
                 options=[{'label': 'Five', 'value': '5'},
                          {'label': 'Ten', 'value': '10'},
                          {'label': 'Fifteen', 'value': '15'},
                          {'label': 'Thirty', 'value': '30'}],
                 value='5'),
    html.Img(id='feature-importance-output', style={'height': '30%', 'width': '40%'})
])

app.layout = html.Div([
    html.H1('Cloud Computing Final Project'),
    dcc.Tabs(id="tabs-example-graph", value='pca-plot',
             children=[
                 dcc.Tab(label='PCA', value='pca-plot', children=tab1),
                 dcc.Tab(label='Validation Curve', value='validation-curve', children=tab2),
                 dcc.Tab(label='Learning Curve', value='learning-curve', children=tab3),
                 dcc.Tab(label='Elbow Plot', value='elbow-plot', children=tab4),
                 dcc.Tab(label='Silhouette Plot', value='silhouette-plot', children=tab5),
                 dcc.Tab(label='Class Imbalance Plot', value='class-imbalance-plot', children=tab6),
                 # dcc.Tab(label='Residuals Plot', value='residuals-plot', children=tab7),
                 # dcc.Tab(label='Prediction Error Plot', value='prediction-error-plot', children=tab8),
                 dcc.Tab(label='Box Plot', value='outlier-plot', children=tab9),
                 dcc.Tab(label='Feature Importance Plot', value='feature-importance-plot', children=tab10),
             ]),
])


def render_content(tab):
    if tab == 'pca-plot':
        return tab1
    elif tab == 'validation-curve':
        return tab2
    elif tab == 'learning-curve':
        return tab3
    elif tab == 'elbow-plot':
        return tab4
    elif tab == 'silhouette-plot':
        return tab5
    elif tab == 'class-imbalance-plot':
        return tab6
    # elif tab == 'residuals-plot':
    #     return tab7
    # elif tab == 'prediction-error-plot':
    #     return tab8
    elif tab == 'outlier-plot':
        return tab9
    elif tab == 'feature-importance-plot':
        return tab10


@app.callback(Output('pca-output', 'src'),
              Input('pca-input', 'value'))
def render_pca_plot(input_value):
    if input_value is None:
        pass
    else:
        image_dir = 'assets/'
        image_path = image_dir + str(input_value) + 'D_PCA_Plot.png'
        return image_path


@app.callback(Output('validation-curve-output', 'src'),
              Input('validation-curve-input', 'value'))
def render_validation_curve(input_value):
    if input_value is None:
        pass
    else:
        image_dir = 'assets/'
        image_path = image_dir + 'validation_curve_' + input_value + '.png'
        return image_path


@app.callback(Output('learning-curve-output', 'src'),
              Input('learning-curve-input', 'value'))
def render_learning_curve(input_value):
    if input_value is None:
        pass
    else:
        image_dir = 'assets/'
        image_path = image_dir + 'learning_curve_' + input_value + '.png'
        return image_path


@app.callback(Output('silhouette-plot-output', 'src'),
              Input('silhouette-plot-input', 'value'))
def render_silhouette_plot(input_value):
    if input_value is None:
        pass
    else:
        image_dir = 'assets/'
        image_path = image_dir + 'silhouette_plot_k' + input_value + '.png'
        return image_path


@app.callback(Output('outlier-output', 'figure'),
              Input('outlier-input', 'value'))
def render_box_plot(input_value):
    data = pd.read_csv(data_dir + 'application_train.csv',
                       usecols=['TARGET', input_value])
    fig = px.box(data, x=input_value)
    return fig


@app.callback(Output('feature-importance-output', 'src'),
              Input('feature-importance-input', 'value'))
def render_feature_importance_plot(input_value):
    if input_value is None:
        pass
    else:
        image_path = 'assets/feature_importance_plot' + input_value + '.png'
        return image_path


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
app.config.suppress_callback_exceptions = True
