# This file runs the epidemic model simulation.

import os
from datetime import date, datetime, timedelta
from typing import Callable, Type, TypeVar

from models import SIR_Model
from plot import plot_US_forecast, plot_global_forecast, plot_timeseries, plot_training_cost
from data_utils import CAN_COUNTY_JSON_DATAPATH, CAN_STATE_JSON_DATAPATH, CAN_COVID_Data, OWID_COVID_Data, CAN_US_COUNTY_URL, CAN_US_STATE_URL

UPDATE = False

# Model
MODEL = SIR_Model
DATATYPE = SIR_Model.get_datatype()

# Formatting
DATE_INPUT_FORMAT = "%m/%d/%Y"

# Training parameters
LEARN_RATE = 300
MAX_STEPS = 100
TOLERANCE = 0

# Defaults
DEFAULT_LOC_TYPE = "state"
DEFAULT_STATE = "WA"
DEFAULT_R0 = 3.0
DEFAULT_INFECTIOUS_PERIOD = 5.1
DEFAULT_START_DATE = "1/1/2022" # "6/1/2021"
DEFAULT_END_DATE = date.today().strftime(DATE_INPUT_FORMAT)
DEFAULT_FORECAST_LEN = 50
DEFAULT_DATA_WEIGHT = 1.0
DEFAULT_TRAINING_WEIGHT = 0.1

# Gets valid user input using the prompt, expecting given type. Continues to 
# request user input until a valid response is received. The value
# will also return true when passed to the valid function.
#
# Parameters:
#   - prompt: prompt to give user.
#   - default: default value for empty response.
#   - type: type of input expected.
#   - is_valid: function for determining if input is valid.
# Returns:
#   The valid value received from user input.
TInput = TypeVar('TInput')
def get_valid_input(prompt: str, default: TInput, type: Type[TInput],
  is_valid: Callable[[TInput], bool]) -> TInput:

  while True:
    inputString = input(prompt)
    if not inputString and default:
      print("Using default: " + str(default))
      return default
    try:
      val = type(inputString)
      if not is_valid(val):
        raise ValueError
      return val
    except ValueError:
      print("Oops! '{}' was not a valid input. Try again...".format(inputString))

# Evaluates if the given string is in a valid date format.
#
# Parameters:
#   - val: the string to be evaluated
# Returns:
#   True iff val is of valid format, matching DATE_INPUT_FORMAT.
def is_valid_date(val: str):
  try:
    datetime.strptime(val, DATE_INPUT_FORMAT).date()
    return True
  except ValueError:
    return False

# Initializes data from user input.
def init_data():
  global epidemic_data, loc_type

  while True:
    try:
      loc_type = get_valid_input("Do you want to analyze county, state, or country data? ",
        DEFAULT_LOC_TYPE, str, lambda val : val == 'county' or val == 'state' or val == 'country')

      if loc_type == 'county':
        os.makedirs("data/county", exist_ok=True)
        state = get_valid_input("From which state would you like to analyze counties? [i.e. WA] ",
          DEFAULT_STATE, str, lambda val : len(val) == 2 and val.isalpha()).upper()
        epidemic_data = CAN_COVID_Data('county', state=state, update=UPDATE)
      elif loc_type == 'country':
        epidemic_data = OWID_COVID_Data(UPDATE)
      else:
        epidemic_data = CAN_COVID_Data('state', update=UPDATE)
      break
    except ValueError as err:
      print(err)

# Gets parameters from user input.
def get_parameters():
  global loc, start_date, end_date, forecast_len
  global r0, infectious_period, data_weight, training_weight

  loc = get_valid_input("Which {} do you want to analyze? ".format(loc_type),
      next(iter(epidemic_data.get_locations())), str, lambda val : val in epidemic_data.get_locations())
  r0 = get_valid_input("What initial R0 value would you like to use? [Positive Float] ",
    DEFAULT_R0, float, lambda val : val > 0)
  infectious_period = get_valid_input("What initial infectious period would you like to use? [Positive Float] ",
    DEFAULT_INFECTIOUS_PERIOD, float, lambda val : val > 0)
  start_date_str = get_valid_input("Which date would you like to start analysis from? [mm/dd/yyyy] ",
    DEFAULT_START_DATE, str, is_valid_date)
  start_date = datetime.strptime(start_date_str, DATE_INPUT_FORMAT).date()
  end_date_str = get_valid_input("Which date would you like to end analysis? [mm/dd/yyyy] ",
    DEFAULT_END_DATE, str, is_valid_date)
  end_date = datetime.strptime(end_date_str, DATE_INPUT_FORMAT).date()
  forecast_len = get_valid_input("How many days would you like to forecast? [Integer] ",
    DEFAULT_FORECAST_LEN, int, lambda val : val > 0)
  data_weight = get_valid_input("How much weight do you give the data? [Float, 0 to 1] ",
    DEFAULT_DATA_WEIGHT, float, lambda val : val >= 0 and val <= 1)  
  training_weight = get_valid_input("What training weight would you like to use? [Float, 0 to 1] ",
    DEFAULT_TRAINING_WEIGHT, float, lambda val : val >= 0 and val <= 1)  

# Simulates epidemic for given parameters.
def simulate():
  global start_date

  loc_to_model_ts = {}
  min_end_date = end_date # last date for which all locations have data
  for l in epidemic_data.get_locations():
    print("Analyzing location {}...".format(l))
    model = MODEL(r0 / infectious_period, 1 / infectious_period)

    data_ts = epidemic_data.get_timeseries(l, start_date, end_date, round(infectious_period), DATATYPE)
    if not len(data_ts.keys()):
      print("Insufficient data for location {}, skipping...".format(l))
      continue

    loc_start_date = min(data_ts.keys())
    loc_end_date = max(data_ts.keys())
    min_end_date = min(min_end_date, loc_end_date)

    costs = model.train_model(data_ts, loc_start_date, loc_end_date, data_weight, LEARN_RATE, MAX_STEPS, TOLERANCE)
    model_ts = model.get_timeseries(data_ts[loc_end_date], loc_end_date, loc_end_date+timedelta(days=forecast_len))

    if (l == loc):
      print("Final gamma: {}".format(model.gamma))
      print("Final beta: {}".format(model.beta))
      plot_timeseries(DATATYPE, model_ts, data_ts, loc)
      plot_training_cost(DATATYPE, costs, loc)

    for dt, data in model_ts.items():
      data_ts[dt] = data # add model to data
    loc_to_model_ts[l] = data_ts

  print("Plotting data on map...")
  if loc_type == 'county':
    pass # TODO
  elif loc_type == 'country':
    plot_global_forecast(DATATYPE, loc_to_model_ts, min_end_date)
  else:
    plot_US_forecast(DATATYPE, loc_to_model_ts, min_end_date)


# Main execution.
if __name__ == "__main__":
  init_data()
  while True:
    get_parameters()
    simulate()
    resp = input("Do you want to continue? [Y/n] ")
    if not resp == 'Y' and not resp == 'y':
      break
    resp = input("Would you like to analyze different data? [Y/n] ")
    if resp == 'Y' or resp == 'y':
      init_data()
    
