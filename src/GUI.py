import json
from tarfile import NUL
from dash import Dash, html, dcc, Input, Output, callback_context
from src.Heatmap import Heatmap

app = Dash(__name__)
df = NUL
parameter = 'FULLVAL'

class GUI():
    def __init__(self, d_f):
        global df
        df = d_f
        self.app = app

    def run(self):
        global parameter
        heatmap = Heatmap(df, parameter)
        self.app.layout = html.Div(style={
            'textAlign': 'center'
        },
        children=[
            html.H1(children='TECHNOPOL-AI'),

            html.Label('Choose parameter'),

            html.Div([
                dcc.Dropdown(df.columns, parameter, style={'width': 250, 'margin-right': 'auto', 'margin-left': 'auto'}
                    , id='parameter-dropdown'),

                dcc.Checklist(
                    ['3D View'],
                    [],
                    id='3d-check'
                )
            ]),

            html.Br(),

            html.Div(
                    html.Div(
                        dcc.Graph(
                            id="New York Graph",
                            figure=heatmap.get_2d_plot(),
                            responsive=True,
                            style={
                                "width": "100%",
                                "height": "100%"
                            }
                        ),
                        style={
                            "width": "100%",
                            "height": "100%",
                        },
                    ),
                    id='graph-div',
                    style={
                            "width": "68%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        }
                )
        ])
        self.app.run_server(debug=True)

    @app.callback(
        Output('graph-div', 'children'),
        Input('parameter-dropdown', 'value'),
        Input('3d-check', 'value')
    )
    def update_output(parameter_value, check_value):
        global parameter
        parameter = parameter_value

        heatmap = Heatmap(df, parameter)
        if '3D View' in check_value:
            fig = heatmap.get_3d_plot()
        else:
            fig=heatmap.get_2d_plot()

        return dcc.Graph(
            id="New York Graph",
            figure=fig,
            responsive=True,
            style={
                "width": "100%",
                "height": "100%"
            }
        )




