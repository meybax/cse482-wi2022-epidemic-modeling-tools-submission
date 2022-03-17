# This file contains the dash application for the project

import dash
import logging
import diskcache
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Callable, Dict, List, Tuple, Type
from datetime import date, timedelta
from dash import Dash, html, dcc, Input, Output, State, no_update
from dash.long_callback import DiskcacheLongCallbackManager
from data_utils import CAN_COVID_Data, Epidemic_Data, OWID_COVID_Data

from models import Epidemic_Model, SIR_Model, TEpidemic_Model, TModel_Data
from plot import get_chloropleth_figure, get_df_from_ts

DATA_COLOR_SCALE = px.colors.sequential.Bluered
FORECAST_COLOR_SCALE = px.colors.sequential.Plasma

# Model names as string
SIR_MODEL_STR = 'SIR Model'

# Default model analysis parameters
DEFAULT_MODEL = SIR_Model
DEFAULT_MODEL_NAME = SIR_MODEL_STR
DEFAULT_R0 = 3.0
DEFAULT_INFECTIOUS_PERIOD = 5.1
DEFAULT_PARAMS = (DEFAULT_R0 / DEFAULT_INFECTIOUS_PERIOD, 1 / DEFAULT_INFECTIOUS_PERIOD)
DEFAULT_START_DATE = date(2022, 1, 1)
DEFAULT_END_DATE = date.today()
DEFAULT_FORECAST_START = DEFAULT_END_DATE
DEFAULT_FORECAST_LEN = 50

# Default training parameters
DEFAULT_DATA_WEIGHT = 1.0
DEFAULT_LEARN_RATE = 300
DEFAULT_MAX_STEPS = 10
DEFAULT_TOLERANCE = 0

def get_timeseries_data(model_type: Type[TEpidemic_Model], epidemic_data: Epidemic_Data,
    loc_to_model_ts: Dict[str, Dict[date, TModel_Data]], loc_to_data_ts: Dict[str, Dict[date, TModel_Data]], loc_to_costs: Dict[str, List[TModel_Data]], 
    params: Tuple, infectious_period: int, start_date: date, end_date: date, forecast_start: date, forecast_len: int,
    data_weight: float, learn_rate: float, max_steps: int, tolerance: float, set_progress: Callable):

    data_type = model_type.get_datatype()
    iter = 0
    num_locations = len(epidemic_data.get_locations())
    set_progress((str(iter), str(num_locations)))
    last_date_with_data = start_date
    for l in epidemic_data.get_locations():
        print("Analyzing location {}...".format(l))
        model = model_type.init(params)

        data_ts = epidemic_data.get_timeseries(l, start_date, end_date, infectious_period, data_type)
        if not len(data_ts.keys()):
            print("Insufficient data, skipping.")
            continue

        country_start_date = min(data_ts.keys())
        country_end_date = max(data_ts.keys())
        last_date_with_data = max(last_date_with_data, country_end_date)
        forecast_analysis_start = max(country_start_date, min(country_end_date, forecast_start))

        try:
            costs = model.train_model(data_ts, country_start_date, country_end_date, data_weight, learn_rate, max_steps, tolerance)
        except OverflowError:
            print("Training failed, using default parameters.")
            costs = []
            model = model_type.init(params)
        model_ts = model.get_timeseries(data_ts[forecast_analysis_start], forecast_analysis_start, forecast_start+timedelta(days=forecast_len))

        loc_to_model_ts[l] = model_ts
        loc_to_data_ts[l] = data_ts
        loc_to_costs[l] = costs

        # Update progress
        iter += 1
        set_progress((str(iter), str(num_locations)))

    return min(forecast_start, last_date_with_data)

def get_model(model_str: str) -> Type[Epidemic_Model]:
    if model_str == SIR_MODEL_STR:
        return SIR_Model
    raise NotImplementedError

def get_date_slider_marks(start: date, end: date) -> Dict[int, str]:
    result = {}
    result[0] = str(start.isoformat())
    result[(end-start).days] = str(end.isoformat())

    i = 0
    curr_date = start + timedelta(days=10)
    while curr_date <= end - timedelta(days=10):
        if curr_date.day == 1:
            result[(curr_date-start).days] = str(curr_date.strftime("%Y-%m"))
        curr_date += timedelta(days=1)
    
    return result

def update_graph_with_slider(slider_val: int, model_type_val: str, start_date_val: str, forecast_start_val: str,
    model_df_json: str, data_df_json: str, loc_type: str, locationmode: str, projection: str, title: str):

    model_type = get_model(model_type_val)

    forecast_start = date.fromisoformat(forecast_start_val)
    start_date = date.fromisoformat(start_date_val)
    curr_date = start_date + timedelta(days=slider_val)

    if curr_date > forecast_start:
        df = pd.read_json(model_df_json, orient='table')
        color_scale = FORECAST_COLOR_SCALE
        title = "Forecasted " + title
    else:
        df = pd.read_json(data_df_json, orient='table')
        color_scale = DATA_COLOR_SCALE

    df['date'] = df['date'].dt.date
    fig = get_chloropleth_figure(model_type.get_datatype(), df[(df['date'] == curr_date)], color_scale, 
        loc_type, locationmode, projection, title.format(curr_date.isoformat()))

    return fig

############################## DASH APPLICATION ##############################

# Init application
logging.getLogger('werkzeug').setLevel(logging.ERROR)
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, long_callback_manager=long_callback_manager, external_stylesheets=external_stylesheets)

styles = {
    'visible-progress': {
        "visibility": "visible",
        "width": "30%",
        "margin-left": "auto",
        "margin-right": "auto",
        "display": "block"}
}

def graph_container(loc_type):
    return html.Div(id="{}-graph-container".format(loc_type), children=[
        dcc.Store(id="{}-model-df".format(loc_type)),
        dcc.Store(id="{}-data-df".format(loc_type)),
        dcc.Store(id="{}-costs-df".format(loc_type)),

        html.Div([
            html.Div(children=[
                dcc.Graph(
                    id="{}-graph".format(loc_type),
                    figure={}
                ),
            ], style={'width': '59%', 'display': 'inline-block', 'padding': '0 20'}),

            html.Div(children=[
                html.Div(id="{}-plots".format(loc_type), style={"display":"block"}),
                html.Div(id="{}-cost-plots".format(loc_type), style={"display":"none"}),
            ], style={'display': 'inline-block', 'width': '39%', 'height': '400px', 'overflow-y': 'auto'}),
        ]),

        html.Div([
            html.Div(dcc.Slider(
                    id="{}-date-slider".format(loc_type),
                    min = 0,
                    max = (max(DEFAULT_FORECAST_START + timedelta(days=DEFAULT_FORECAST_LEN), DEFAULT_END_DATE) \
                        - DEFAULT_START_DATE).days,
                    value = (DEFAULT_FORECAST_START-DEFAULT_START_DATE).days,
                    step = 1,
                    marks=get_date_slider_marks(DEFAULT_START_DATE,
                        max(DEFAULT_FORECAST_START + timedelta(days=DEFAULT_FORECAST_LEN), DEFAULT_END_DATE))
            ), style={'display': 'inline-block', 'width': '50%', 'padding': '0px 100px 20px 20px'}),

            html.Div(
                dcc.Dropdown(["Model Data", "Costs"], "Model Data", id="{}-plot-select".format(loc_type), clearable=False),
                style={'display': 'inline-block', 'width': '20%', 'padding': '0px 100px 20px 20px'}
            ),
        ]),
    ], style={'display': 'none'})

def graph_loading(loc_type):
    return html.Div(id="{}-graph-loading".format(loc_type), children=[
        dcc.Loading(
            id="{}-loading".format(loc_type),
            type="default",
            children=dcc.Store(id="{}-data-signal".format(loc_type)),
        ),
        html.Progress(id="{}-progress-bar".format(loc_type), style={"visibility": "hidden"}),
    ])

app.layout = html.Div(children=[
    html.H1(children='Epidemic Modeling Tools'),

    html.H2(children='COVID-19 Epidemic Modeling'),

    html.Div(id="model-parameters", children=[
        dcc.Markdown("""
            **Model Parameters**
        """),
        dcc.Markdown("""
            Select a model to use for modeling the epidemic.
        """),
        dcc.Dropdown([SIR_MODEL_STR], DEFAULT_MODEL_NAME, id="model-input", clearable=False),
        dcc.Markdown("""
            Input the initial average infectious period to use. (Positive Float)
        """),
        dcc.Input(id="infectious-period-input", type="number", min=0, value=DEFAULT_INFECTIOUS_PERIOD),
        html.Div(id='sir-model-params', children=[
            dcc.Markdown("""
                Input the initial R0 value to use. (Positive Float)
            """),
            dcc.Input(id="r0-input", type="number", min=0, value=DEFAULT_R0),
        ], style={'display': 'none'}),
    ]), html.Br(),

    html.Div(id="analysis-dates", children=[
        dcc.Markdown("""
            **Analysis Dates**
        """),

        html.Div(children=[
            dcc.Markdown("""
                Select a start date for data analysis.
            """),
            dcc.DatePickerSingle(
                id='start-date-input',
                min_date_allowed=date(2019, 9, 1),
                max_date_allowed=date.today(),
                initial_visible_month=DEFAULT_START_DATE,
                date=DEFAULT_START_DATE
            ),
        ], style={'display': 'inline-block', 'width': '20%'}),

        html.Div(children=[
            dcc.Markdown("""
                Select an end date for data analysis.
            """),
            dcc.DatePickerSingle(
                id='end-date-input',
                min_date_allowed=DEFAULT_START_DATE,
                max_date_allowed=date.today(),
                initial_visible_month=DEFAULT_END_DATE,
                date=DEFAULT_END_DATE
            ),
        ], style={'display': 'inline-block', 'width': '20%'}),

        html.Div(children=[
            dcc.Markdown("""
                Select a start date for the forecast.
            """),
            dcc.DatePickerSingle(
                id='forecast-start-input',
                min_date_allowed=DEFAULT_START_DATE,
                max_date_allowed=DEFAULT_END_DATE,
                initial_visible_month=DEFAULT_FORECAST_START,
                date=DEFAULT_FORECAST_START
            ),
        ], style={'display': 'inline-block', 'width': '20%'}),

        html.Div(children=[
            dcc.Markdown("""
                Input the number of days to forecast. (Positive Integer)
            """),
            dcc.Input(id="forecast-len-input", type="number", min=0, step=1, value=DEFAULT_FORECAST_LEN),
        ], style={'display': 'inline-block', 'width': '30%'}),
    ]), html.Br(),

    html.Div(id="Training Parameters", children=[
        dcc.Markdown("""
            **Training Parameters**
        """),

        html.Div(children=[
            dcc.Markdown("""
                Input the weight to give the data.
                
                (Float, between 0 and 1)
            """),
            dcc.Input(id="data-weight-input", type="number", min=0, max=1, value=DEFAULT_DATA_WEIGHT),
        ], style={'display': 'inline-block', 'width': '22%'}),

        html.Div(children=[
            dcc.Markdown("""
                Input the rate that the model should learn.
                
                (Positive Float)
            """),
            dcc.Input(id="learn-rate-input", type="number", min=0, value=DEFAULT_LEARN_RATE),
        ], style={'display': 'inline-block', 'width': '22%'}),

        html.Div(children=[
            dcc.Markdown("""
                Input the maximum number of iterations to train the model.
                
                (Positive Integer)
            """),
            dcc.Input(id="max-steps-input", type="number", min=0, step=1, value=DEFAULT_MAX_STEPS),
        ], style={'display': 'inline-block', 'width': '22%'}),

        html.Div(children=[
            dcc.Markdown("""
                Input the minimum cost tolerance before finishing training.
                
                (Positive Float)
            """),
            dcc.Input(id="tolerance-input", type="number", min=0, value=DEFAULT_TOLERANCE),
        ], style={'display': 'inline-block', 'width': '22%'}),
    ]), html.Br(),

    html.Div(children=[
        html.Button('Analyze', id='analyze_button_id', n_clicks=0),
        html.Button(id="cancel_button_id", children="Cancel")
    ]),

    graph_container('country'),
    graph_loading('country'),
    html.Br(),

    graph_container('state'),
    graph_loading('state'),
    html.Br(),

    dcc.Store(id="model-type-val"),
    dcc.Store(id="infectious-period-val"),
    dcc.Store(id="r0-val"),

    dcc.Store(id="start-date-val"),
    dcc.Store(id="end-date-val"),
    dcc.Store(id="forecast-start-val"),
    dcc.Store(id="forecast-len-val"),

    dcc.Store(id="data-weight-val"),
    dcc.Store(id="learn-rate-val"),
    dcc.Store(id="max-steps-val"),
    dcc.Store(id="tolerance-val"),
])

@app.callback(
    Output("sir-model-params", "style"),
    Input("model-input", "value"),
)
def update_model_param_display(model_input):
    if model_input == SIR_MODEL_STR:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output("end-date-input", "min_date_allowed"),
    Output("forecast-start-input", "min_date_allowed"),
    Input("start-date-input", "date"),
)
def update_daterange_start(start_date_input):
    if start_date_input is None:
        return no_update, no_update
    start_date = date.fromisoformat(start_date_input)
    return start_date, start_date

@app.callback(
    Output("forecast-start-input", "max_date_allowed"),
    Output("forecast-start-input", "initial_visible_month"),
    Output("forecast-start-input", "date"),
    Input("end-date-input", "date"),
)
def update_daterange_end(end_date_input):
    if end_date_input is None:
        return no_update, no_update, no_update
    end_date = date.fromisoformat(end_date_input)
    return end_date, end_date, end_date

####################### COUNTRY GRAPH CALLBACKS #######################

@app.callback(
    Output('country-data-signal', 'data'),
    Input("analyze_button_id", "n_clicks"),
    State('country-data-signal', 'data')
)
def update_country_data(clicks, data_signal):
    if not clicks:
        return no_update
    if data_signal is None or date.fromisoformat(data_signal) != date.today():
        OWID_COVID_Data(update=True)
    return date.today().isoformat()

@app.long_callback(
    output=[
        Output("country-graph-container", "style"),
        Output("country-graph", "figure"),

        Output("model-type-val", "data"),
        Output("infectious-period-val", "data"),
        Output("r0-val", "data"),

        Output("start-date-val", "data"),
        Output("end-date-val", "data"),
        Output("forecast-start-val", "data"),
        Output("forecast-len-val", "data"),

        Output("data-weight-val", "data"),
        Output("learn-rate-val", "data"),
        Output("max-steps-val", "data"),
        Output("tolerance-val", "data"),

        Output("country-model-df", "data"),
        Output("country-data-df", "data"),
        Output("country-costs-df", "data"),

        Output("country-date-slider", "max"),
        Output("country-date-slider", "value"),
        Output("country-date-slider", "marks"),
    ],
    inputs=[
        Input("country-data-signal", "data"),
        Input("country-date-slider", "value"),

        State("model-input", "value"),
        State("infectious-period-input", "value"),
        State("r0-input", "value"),

        State("start-date-input", "date"),
        State("end-date-input", "date"),
        State("forecast-start-input", "date"),
        State("forecast-len-input", "value"),

        State("data-weight-input", "value"),
        State("learn-rate-input", "value"),
        State("max-steps-input", "value"),
        State("tolerance-input", "value"),

        State("model-type-val", "data"),
        State("start-date-val", "data"),
        State("forecast-start-val", "data"),

        State("country-model-df", "data"),
        State("country-data-df", "data"),
    ],
    running=[
        (Output("analyze_button_id", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
        (
            Output("country-progress-bar", "style"),
            styles["visible-progress"],
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=[Output("country-progress-bar", "value"), Output("country-progress-bar", "max")],
)
def update_country_graph(set_progress, data_signal, slider_val,
    model_input, infectious_period_input, r0_input,
    start_date_input, end_date_input, forecast_start_input, forecast_len_input,
    data_weight_input, learn_rate_input, max_steps_input, tolerance_input,
    model_type_val, start_date_val, forecast_start_val, country_model_df_json, country_data_df_json):

    ctx = dash.callback_context

    default_ret = (no_update,)*19
    if not ctx.triggered:
        return default_ret

    # Process callback for slider input
    if ctx.triggered[0]['prop_id'].split('.')[0] == "country-date-slider":
        fig = update_graph_with_slider(slider_val, model_type_val, start_date_val, forecast_start_val,
            country_model_df_json, country_data_df_json,
            'country', 'country names', 'robinson', "Infections per 100K People on {}")
        return default_ret[:1] + (fig,) + default_ret[2:]

    # Return if not triggered by data signal or data is out of date
    if not ctx.triggered[0]['prop_id'].split('.')[0] == "country-data-signal" \
            or date.fromisoformat(data_signal) != date.today():
        return default_ret

    country_epidemic_data = OWID_COVID_Data(update=False)

    # Get model parameters
    model_type = get_model(model_input)
    params = DEFAULT_PARAMS
    if not infectious_period_input:
        return default_ret
    infectious_period = round(infectious_period_input)
    if model_type == SIR_Model:
        if not r0_input:
            return default_ret
        params = (r0_input/infectious_period, 1/infectious_period)
    
    # Get analysis dates
    if start_date_input is None:
        return default_ret
    start_date = date.fromisoformat(start_date_input)
    if end_date_input is None:
        return default_ret
    end_date = date.fromisoformat(end_date_input)
    if forecast_start_input is None:
        return default_ret
    forecast_start = date.fromisoformat(forecast_start_input)
    if forecast_len_input is None:
        return default_ret
    forecast_len = forecast_len_input
    max_date = max(end_date, forecast_start+timedelta(days=forecast_len))

    # Get training parameters
    if data_weight_input is None:
        return default_ret
    data_weight = float(data_weight_input)
    if learn_rate_input is None:
        return default_ret
    learn_rate = float(learn_rate_input)
    if max_steps_input is None:
        return default_ret
    max_steps = max_steps_input
    if tolerance_input is None:
        return default_ret
    tolerance = float(tolerance_input)

    # Run model
    country_to_model_ts = {}
    country_to_data_ts = {}
    country_to_costs = {}
    forecast_start = get_timeseries_data(model_type, country_epidemic_data, country_to_model_ts, country_to_data_ts, country_to_costs,
        params, infectious_period, start_date, end_date, forecast_start, forecast_len,
        data_weight, learn_rate, max_steps, tolerance, set_progress)
    if not len(country_to_data_ts.items()):
        return default_ret

    # Convert costs to dataframe
    costs_dict = {'country': [], 'iteration': []}
    for country, costs in country_to_costs.items():
        for i in range(len(costs)):
            costs_dict['country'].append(country)
            costs_dict['iteration'].append(i)
            for field in model_type.get_datatype().__annotations__.keys():
                if field not in costs_dict:
                    costs_dict[field] = []
                costs_dict[field].append(costs[i][field])
    costs_df = pd.DataFrame(data=costs_dict)

    # Generate figure
    print("Generating figure...")
    data_df = get_df_from_ts(model_type.get_datatype(), country_to_data_ts, 'country')
    model_df = get_df_from_ts(model_type.get_datatype(), country_to_model_ts, 'country')
    color_scale = DATA_COLOR_SCALE
    fig = get_chloropleth_figure(model_type.get_datatype(), data_df[(data_df['date'] == forecast_start)], color_scale,
        'country', 'country names', 'robinson', "Infections per 100K People on {}".format(forecast_start.isoformat()))

    return {'display':'block'}, fig, \
        model_input, infectious_period_input, r0_input, \
        start_date.isoformat(), end_date.isoformat(), forecast_start.isoformat(), forecast_len, \
        data_weight, learn_rate, max_steps, tolerance, \
        model_df.to_json(orient='table'), data_df.to_json(orient='table'), costs_df.to_json(orient='table'), \
        (max_date-start_date).days, (forecast_start-start_date).days, get_date_slider_marks(start_date, max_date)

@app.callback(
    Output('country-plots', 'style'),
    Output('country-cost-plots', 'style'),
    Input('country-plot-select', 'value')
)
def update_country_plot_display(plot_select):
    if plot_select == "Costs":
        return {"display":"none"}, {"display":"block"}
    else:
        return {"display":"block"}, {"display":"none"}

@app.callback(
    Output('country-plots', 'children'),
    Output('country-cost-plots', 'children'),
    Input('country-graph', 'hoverData'),
    Input("model-type-val", "data"),
    Input("country-model-df", "data"),
    Input("country-data-df", "data"),
    Input("country-costs-df", "data"),
)
def display_country_plots(hover_data, model_type_val, country_model_df_str, country_data_df_str, country_costs_df_str):
    default_ret = no_update, no_update
    if hover_data is None:
        return default_ret

    country = hover_data['points'][0]['location']
    model_type = get_model(model_type_val)

    model_df = pd.read_json(country_model_df_str, orient='table')
    model_df = model_df[(model_df['country'] == country)]
    data_df = pd.read_json(country_data_df_str, orient='table')
    data_df = data_df[(data_df['country'] == country)]
    costs_df = pd.read_json(country_costs_df_str, orient='table')
    costs_df = costs_df[(costs_df['country'] == country)]

    fields = list(model_type.get_datatype().__annotations__.keys())
    fields.remove('population')
    num_graphs = len(fields) + 1

    # Get model data graphs
    data_graphs = list(go.Figure() for _ in range(num_graphs))
    for i, f in enumerate(fields):
        model_plot = go.Scatter(x=model_df['date'], y=model_df[f].str.replace(',', '').astype(int),
            mode='lines', name="{} (Model)".format(f.title()))
        data_plot = go.Scatter(x=data_df['date'], y=data_df[f].str.replace(',', '').astype(int),
            mode='lines', name="{} (Data)".format(f.title()))
        data_graphs[0].add_trace(model_plot)
        data_graphs[0].add_trace(data_plot)
        data_graphs[i+1].add_trace(model_plot)
        data_graphs[i+1].add_trace(data_plot)
        data_graphs[i+1].update_layout(title="Number of {}".format(f.title()))
    data_graphs[0].update_layout(title="{} Data for {}".format(model_type_val, country))

    # Get training cost graphs
    cost_graphs = list(go.Figure() for _ in range(num_graphs))
    for i, f in enumerate(fields):
        plot = go.Scatter(x=costs_df['iteration'], y=costs_df[f],
            mode='lines', name="{}".format(f.title()))
        cost_graphs[0].add_trace(plot)
        cost_graphs[i+1].add_trace(plot)
        cost_graphs[i+1].update_layout(title="Cost for {}".format(f.title()))
    cost_graphs[0].update_layout(title="Training Costs for {}".format(country))

    return list(dcc.Graph(figure=fig) for fig in data_graphs), list(dcc.Graph(figure=fig) for fig in cost_graphs)

####################### STATE GRAPH CALLBACKS #######################

@app.callback(
    Output('state-data-signal', 'data'),
    Input("country-graph", "clickData"),
    State('state-data-signal', 'data')
)
def update_state_data(click_data, data_signal):
    if click_data is None or click_data['points'][0]['location'] != 'United States':
        return no_update
    if data_signal is None or date.fromisoformat(data_signal) != date.today():
        CAN_COVID_Data('state', update=True)
    return date.today().isoformat()

@app.long_callback(
    output=[
        Output("state-graph-container", "style"),
        Output("state-graph", "figure"),

        Output("state-model-df", "data"),
        Output("state-data-df", "data"),
        Output("state-costs-df", "data"),

        Output("state-date-slider", "max"),
        Output("state-date-slider", "value"),
        Output("state-date-slider", "marks"),
    ],
    inputs=[
        Input("state-data-signal", "data"),
        Input("state-date-slider", "value"),
        Input("analyze_button_id", "n_clicks"),

        State("model-type-val", "data"),
        State("infectious-period-val", "data"),
        State("r0-val", "data"),

        State("start-date-val", "data"),
        State("end-date-val", "data"),
        State("forecast-start-val", "data"),
        State("forecast-len-val", "data"),

        State("data-weight-val", "data"),
        State("learn-rate-val", "data"),
        State("max-steps-val", "data"),
        State("tolerance-val", "data"),

        State("state-model-df", "data"),
        State("state-data-df", "data"),
    ],
    running=[
        (
            Output("state-progress-bar", "style"),
            styles["visible-progress"],
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=[Output("state-progress-bar", "value"), Output("state-progress-bar", "max")],
)
def update_state_graph(set_progress, data_signal, slider_val, clicks,
    model_type_val, infectious_period_val, r0_val,
    start_date_val, end_date_val, forecast_start_val, forecast_len_val,
    data_weight_val, learn_rate_val, max_steps_val, tolerance_val,
    state_model_df_json, state_data_df_json):

    ctx = dash.callback_context

    default_ret = (no_update,)*8
    if not ctx.triggered:
        return default_ret

    # Process callback for slider input
    if ctx.triggered[0]['prop_id'].split('.')[0] == "state-date-slider":
        fig = update_graph_with_slider(slider_val, model_type_val, start_date_val, forecast_start_val,
            state_model_df_json, state_data_df_json,
            'state', 'USA-states', 'albers usa', "Infections per 100K People on {}")
        return default_ret[:1] + (fig,) + default_ret[2:]

    if ctx.triggered[0]['prop_id'].split('.')[0] == "analyze_button_id" and clicks:
        return ({"display":"none"},) + default_ret[1:]

    # Return if not triggered by data signal or data is out of date
    if not ctx.triggered[0]['prop_id'].split('.')[0] == "state-data-signal" \
            or date.fromisoformat(data_signal) != date.today():
        return default_ret

    state_epidemic_data = CAN_COVID_Data('state', update=False)

    # Get model parameters
    model_type = get_model(model_type_val)
    params = DEFAULT_PARAMS
    infectious_period = round(float(infectious_period_val))
    if model_type == SIR_Model:
        params = (float(r0_val)/infectious_period, 1/infectious_period)
    
    # Get analysis dates
    start_date = date.fromisoformat(start_date_val)
    end_date = date.fromisoformat(end_date_val)
    forecast_start = date.fromisoformat(forecast_start_val)
    forecast_len = int(forecast_len_val)
    max_date = max(end_date, forecast_start+timedelta(days=forecast_len))

    # Get training parameters
    data_weight = float(data_weight_val)
    learn_rate = float(learn_rate_val)
    max_steps = int(max_steps_val)
    tolerance = float(tolerance_val)

    # Run model
    state_to_model_ts = {}
    state_to_data_ts = {}
    state_to_costs = {}
    forecast_start = get_timeseries_data(model_type, state_epidemic_data, state_to_model_ts, state_to_data_ts, state_to_costs,
        params, infectious_period, start_date, end_date, forecast_start, forecast_len,
        data_weight, learn_rate, max_steps, tolerance, set_progress)
    if not len(state_to_data_ts.items()):
        return default_ret

    # Convert costs to dataframe
    costs_dict = {'state': [], 'iteration': []}
    for state, costs in state_to_costs.items():
        for i in range(len(costs)):
            costs_dict['state'].append(state)
            costs_dict['iteration'].append(i)
            for field in model_type.get_datatype().__annotations__.keys():
                if field not in costs_dict:
                    costs_dict[field] = []
                costs_dict[field].append(costs[i][field])
    costs_df = pd.DataFrame(data=costs_dict)

    # Generate figure
    print("Generating figure...")
    data_df = get_df_from_ts(model_type.get_datatype(), state_to_data_ts, 'state')
    model_df = get_df_from_ts(model_type.get_datatype(), state_to_model_ts, 'state')
    color_scale = DATA_COLOR_SCALE
    fig = get_chloropleth_figure(model_type.get_datatype(), data_df[(data_df['date'] == forecast_start)], color_scale,
        'state', 'USA-states', 'albers usa', "Infections per 100K People on {}".format(forecast_start.isoformat()))

    return {'display':'block'}, fig, \
        model_df.to_json(orient='table'), data_df.to_json(orient='table'), costs_df.to_json(orient='table'), \
        (max_date-start_date).days, (forecast_start-start_date).days, get_date_slider_marks(start_date, max_date)

@app.callback(
    Output('state-plots', 'style'),
    Output('state-cost-plots', 'style'),
    Input('state-plot-select', 'value')
)
def update_state_plot_display(plot_select):
    if plot_select == "Costs":
        return {"display":"none"}, {"display":"block"}
    else:
        return {"display":"block"}, {"display":"none"}

@app.callback(
    Output('state-plots', 'children'),
    Output('state-cost-plots', 'children'),
    Input('state-graph', 'hoverData'),
    Input("model-type-val", "data"),
    Input("state-model-df", "data"),
    Input("state-data-df", "data"),
    Input("state-costs-df", "data"),
)
def display_state_plots(hover_data, model_type_val, state_model_df_str, state_data_df_str, state_costs_df_str):
    default_ret = no_update, no_update
    if hover_data is None:
        return default_ret

    state = hover_data['points'][0]['location']
    model_type = get_model(model_type_val)

    model_df = pd.read_json(state_model_df_str, orient='table')
    model_df = model_df[(model_df['state'] == state)]
    data_df = pd.read_json(state_data_df_str, orient='table')
    data_df = data_df[(data_df['state'] == state)]
    costs_df = pd.read_json(state_costs_df_str, orient='table')
    costs_df = costs_df[(costs_df['state'] == state)]

    fields = list(model_type.get_datatype().__annotations__.keys())
    fields.remove('population')
    num_graphs = len(fields) + 1

    # Get model data graphs
    data_graphs = list(go.Figure() for _ in range(num_graphs))
    for i, f in enumerate(fields):
        model_plot = go.Scatter(x=model_df['date'], y=model_df[f].str.replace(',', '').astype(int),
            mode='lines', name="{} (Model)".format(f.title()))
        data_plot = go.Scatter(x=data_df['date'], y=data_df[f].str.replace(',', '').astype(int),
            mode='lines', name="{} (Data)".format(f.title()))
        data_graphs[0].add_trace(model_plot)
        data_graphs[0].add_trace(data_plot)
        data_graphs[i+1].add_trace(model_plot)
        data_graphs[i+1].add_trace(data_plot)
        data_graphs[i+1].update_layout(title="Number of {}".format(f.title()))
    data_graphs[0].update_layout(title="{} Data for {}".format(model_type_val, state))

    # Get training cost graphs
    cost_graphs = list(go.Figure() for _ in range(num_graphs))
    for i, f in enumerate(fields):
        plot = go.Scatter(x=costs_df['iteration'], y=costs_df[f],
            mode='lines', name="{}".format(f.title()))
        cost_graphs[0].add_trace(plot)
        cost_graphs[i+1].add_trace(plot)
        cost_graphs[i+1].update_layout(title="Cost for {}".format(f.title()))
    cost_graphs[0].update_layout(title="Training Costs for {}".format(state))

    return list(dcc.Graph(figure=fig) for fig in data_graphs), list(dcc.Graph(figure=fig) for fig in cost_graphs)


if __name__ == '__main__':
    app.run_server(debug=True)
