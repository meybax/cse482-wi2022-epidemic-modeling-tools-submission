# This file contains functions for plotting data and model predictions.

import pandas as pd
import plotly.offline as offline
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import date, timedelta
from typing import Any, List, Type, Dict

from data_utils import JHU_DATE_FORMAT
from models import TModel_Data

pd.options.mode.chained_assignment = None  # default='warn'

# Display model and data timeseries data on matplotlib plots.
#
# Parameters:
#   - datatype: which Model_Data datatype to plot.
#   - model_ts: timeseries data from model.
#   - data_ts: timeseries data from raw data.
#   - loc: location corresponding to the given data.
def plot_timeseries(datatype: Type[TModel_Data], 
  model_ts: Dict[date, TModel_Data], data_ts: Dict[date, TModel_Data], loc: str):

  # Plot initializations
  num_graphs = len(datatype.__annotations__.keys())  # add 1 for shared graph, sub 1 for population
  num_cols = (num_graphs + 1) // 2  # Get ceiling of num_graphs / 2

  axs: Any
  fig, axs = plt.subplots(2, num_cols)
  fig.suptitle("Model Data in " + loc)

  # Set x, y axis formats
  for ax in axs.flat:
    # x-axis as date
    formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.MonthLocator()
    ax.xaxis.set_major_locator(locator)

    # y-axis to use readable notation
    formatter1 = mticker.EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    ax.yaxis.set_major_formatter(formatter1)

    ax.set(xlabel="Time (days)", ylabel="Number of People")

  figManager = plt.get_current_fig_manager()
  figManager.resize(*figManager.window.maxsize())

  # Convert data into individual arrays
  model_fields = {}
  data_fields = {}
  for field in datatype.__annotations__.keys():
    model_fields[field] = [model_ts[d][field] for d in model_ts.keys()]
    data_fields[field] = [data_ts[d][field] for d in data_ts.keys()]

  # Plot data points on each subplot, all data on [0, 0]
  idx = 1
  for field in datatype.__annotations__.keys():
    if field == 'population':
      continue
    axs[0, 0].plot(model_ts.keys(), model_fields[field], label="{} (Model)".format(field.title()))
    axs[0, 0].plot(data_ts.keys(), data_fields[field], label="{} (Data)".format(field.title()))
    axs[idx//2, idx%2].plot(model_ts.keys(), model_fields[field], label="{} (Model)".format(field.title()))
    axs[idx//2, idx%2].plot(data_ts.keys(), data_fields[field], label="{} (Data)".format(field.title()))
    idx += 1
  if idx < num_graphs:
    # Delete extra, if it exists
    fig.delaxes(axs[idx/2, idx%2])

  # Display changes
  for ax in axs.flat:
    ax.set_ylim(ymin=0)
    ax.legend()
  plt.show(block=False)

# Display training cost for model data on matplotlib plots.
#
# Parameters:
#   - datatype: which Model_Data datatype to plot.
#   - costs: list of costs at each training iteration
#   - loc: location corresponding to the given data.
def plot_training_cost(datatype: Type[TModel_Data], costs: List[TModel_Data], loc: str):

  # Plot initializations
  num_graphs = len(datatype.__annotations__.keys())  # add 1 for shared graph, sub 1 for population
  num_cols = (num_graphs + 1) // 2  # Get ceiling of num_graphs / 2

  axs: Any
  fig, axs = plt.subplots(2, num_cols)
  fig.suptitle("Training costs in " + loc)

  # Set x, y axis formats
  for ax in axs.flat:
    # y-axis to use readable notation
    formatter1 = mticker.EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    ax.yaxis.set_major_formatter(formatter1)

    ax.set(xlabel="Iteration", ylabel="Cost")

  figManager = plt.get_current_fig_manager()
  figManager.resize(*figManager.window.maxsize())

  # Convert data into individual arrays
  x_vals = list(i for i in range(len(costs)))
  field_costs = {}
  for field in datatype.__annotations__.keys():
    field_costs[field] = list(c[field] for c in costs)

  # Plot data points on each subplot, all data on [0, 0]
  idx = 1
  for field in datatype.__annotations__.keys():
    if field == 'population':
      continue
    axs[0, 0].plot(x_vals, field_costs[field], label="{} Cost".format(field.title()))
    axs[idx//2, idx%2].plot(x_vals, field_costs[field], label="{} Cost".format(field.title()))
    idx += 1
  if idx < num_graphs:
    # Delete extra, if it exists
    fig.delaxes(axs[idx/2, idx%2])

  # Display changes
  for ax in axs.flat:
    ax.legend()
  plt.show(block=False)

# Returns dataframe for plotly from a timeseries of model data objects.
#
# Parameters:
#   - datatype: the type of model data.
#   - loc_to_model_ts: a map from each location to their respective data.
#   - loc_type: specifies type of location (i.e. state).
# Returns:
#   A Pandas dataframe with the location, date, infection rate, and other data fields.
def get_df_from_ts(datatype: Type[TModel_Data], loc_to_model_ts: Dict[str, Dict[date, TModel_Data]], loc_type: str) -> Any:
  # convert model timeseries to dataframe
  dts = {loc_type: [], 'date': [], 'infection rate': []}
  for loc, model_ts in loc_to_model_ts.items():
    dts[loc_type] += [loc] * len(model_ts)
    dts['date'] += list(model_ts.keys())
    dts['infection rate'] += [model_ts[d]['infected']/model_ts[d]['population']*100000
      for d in model_ts.keys()]
    for field in datatype.__annotations__.keys():
      if not field in dts:
        dts[field] = []
      dts[field] += [model_ts[d][field] for d in model_ts.keys()]
  
  # create data frame, and format large numbers with commas
  df = pd.DataFrame(data=dts)
  for field in datatype.__annotations__.keys():
    df[field] = df[field].astype(int).apply(lambda x : "{:,}".format(x))
  return df

# Plots the state case data on a United States map.
#
# Parameters:
#   - datatype: which Model_Data datatype to plot.
#   - state_to_model_ts: dict mapping location to model_ts dictionary.
#   - epidemic_data: data structure containing information on epidemic.
#   - forecast_start_date: date where forecast begins.
def plot_US_forecast(datatype: Type[TModel_Data], state_to_model_ts: Dict[str, Dict[date, TModel_Data]],
  forecast_start_date: date):

  plot_forecast_helper(datatype, state_to_model_ts, forecast_start_date,
    'state', 'USA-states', 'Infection Rate By State', dict(scope='usa', projection={'type': 'albers usa'}),
    'html/map_us_infected_slider.html')

# Plots the country case data on a global map.
#
# Parameters:
#   - datatype: which Model_Data datatype to plot.
#   - state_to_model_ts: dict mapping location to model_ts dictionary.
#   - epidemic_data: data structure containing information on epidemic.
#   - forecast_start_date: date where forecast begins.
def plot_global_forecast(datatype: Type[TModel_Data], country_to_model_ts: Dict[str, Dict[date, TModel_Data]],
  forecast_start_date: date):

  plot_forecast_helper(datatype, country_to_model_ts, forecast_start_date,
    'country', 'country names', 'Infection Rate By Country', dict(scope='world', projection={'type': 'robinson'}),
    'html/map_global_infected_slider.html')

# Helper to plot forecast for varying locations.
#
# Parameters:
#   - datatype: which Model_Data datatype to plot.
#   - loc_to_model_ts: dict mapping location to model_ts dictionary.
#   - epidemic_data: data structure containing information on epidemic.
#   - forecast_start_date: date where forecast begins.
#   - loc_type: should be "country", "state", or "county",
#   - locationmode: which format to interpret location modes, used by Plotly.
#   - title: title to give plot.
#   - geo: geography dictionary, used for Plotly layout.
#   - filename: name of file to store plot HTML file.
def plot_forecast_helper(datatype: Type[TModel_Data], loc_to_model_ts: Dict[str, Dict[date, TModel_Data]],
  forecast_start_date: date,
  loc_type: str, locationmode: str, title: str, geo: dict, filename: str):

  print("Preparing data into plot format...")
  df = get_df_from_ts(datatype, loc_to_model_ts, loc_type)

  # normal color-scale
  scl = [[0.0, '#69B34C'],[0.2, '#ACB334'],[0.4, '#FAB733'],
        [0.6, '#FF8E15'],[0.8, '#FF4E11'],[1.0, '#FF0D0D']] # green to red

  forecast_scl = [[0.0, '#FFD014'],[0.2, '#DFB12F'],[0.4, '#BF914A'],
                  [0.6, '#A07265'],[0.8, '#805280'],[1.0, '#60339B']] # yellow to purple

  print("Generating data slider...")
  data_slider = []
  for dt in df['date'].unique():
    # get data for a single date
    df_segmented = df[(df['date'] == dt)]

    for col in df_segmented.columns:
      df_segmented[col] = df_segmented[col].astype(str)

    # set hover text
    df_segmented['text'] = ""
    for field in datatype.__annotations__.keys():
      df_segmented['text'] += "<br>{}: ".format(field.title()) + df_segmented[field]

    # set data in day
    data_each_day = dict(
      type='choropleth',
      locations = df_segmented[loc_type],
      z=df_segmented['infection rate'].astype(float),
      locationmode=locationmode,
      colorscale = scl,
      colorbar= {'title':'Infections per 100K'},
      zmax=df.mean(numeric_only=True)['infection rate']
        + df.std(numeric_only=True)['infection rate'],
      zmin=0,
      text=df_segmented['text'] # hover text
    )

    # update color scale if we are forecasting data
    if dt > forecast_start_date:
      data_each_day["colorscale"] = forecast_scl
      data_each_day["colorbar"] = {'title':'FORECAST<br>Infections per 100K'}

    data_slider.append(data_each_day)

  print("Defining steps...")
  steps = []
  for i in range(len(data_slider)):
      step: Dict[Any, Any] = dict(method='restyle',
                  args=['visible', [False] * len(data_slider)],
                  label='Date {}'.format((timedelta(i) + min(df['date'])).strftime(JHU_DATE_FORMAT)))
      step['args'][1][i] = True
      steps.append(step)

  sliders = [dict(active=0, pad={"t": 3}, steps=steps)]

  layout = dict(title=title, geo=geo, sliders=sliders)

  print("Generating plot...")
  fig = dict(data=data_slider, layout=layout)
  offline.plot(fig, auto_play=False, auto_open=True, filename=filename, validate=True)

# Creates a plotly chloropleth figure for the given data on the given day.
#
# Parameters:
#   - datatype: datatype of data to plot.
#   - df: dataframe of data to plot.
#   - curr_date: date to plot.
#   - loc_type: type of location (i.e. state).
#   - locationmode: locationmode for plotly graph.
#   - projection: projection for plotly graph.
#   - title: title for plotly graph.
# Returns:
#   A chloropleth figure from plotly express.
def get_chloropleth_figure(datatype: Type[TModel_Data], df: Any, color_scale: List[str],
  loc_type: str, locationmode: str, projection: str, title: str) -> Dict:

  labels = {}
  for col in df.columns:
    labels[col] = col.title()

  locations = df[loc_type]
  if not len(locations):
    locations = ["United States"]

  df['infection rate'] = df['infection rate'].astype(float)
  fig = px.choropleth(df, locations=df[loc_type],
                      locationmode=locationmode,
                      color='infection rate',
                      color_continuous_scale=color_scale,
                      range_color=[
                        0, df.mean(numeric_only=True)['infection rate']
                        + df.std(numeric_only=True)['infection rate']
                      ],
                      hover_name=loc_type,
                      hover_data=list(datatype.__annotations__.keys()),
                      labels=labels,
                      projection=projection,
                      title=title)
  return fig
