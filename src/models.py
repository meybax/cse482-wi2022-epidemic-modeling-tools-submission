# This file defines epidemic models and related data types.

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Generic, List, OrderedDict, Tuple, Type, TypeVar, Dict, TypedDict, Union, cast

#################### DATA TYPES FOR MODELS ####################

# Type variable for model data for type checking. Requires all
# instances of TModel_Data in a given scope to be the same subclass of
# Model_Data.
TModel_Data = TypeVar('TModel_Data', bound='Model_Data')

# A TypedDict definition specifying a dictionary requiring an 'infected' field.
class Model_Data(TypedDict):
  population: float
  infected: float

# An extension of Model_Data adding on 'susceptible' and 'recovered' fields.
class SIR_Data(Model_Data):
  susceptible: float
  recovered: float

# An extension of SIR_Data adding on a 'dead' field.
class SIRD_Data(SIR_Data):
  dead: float

# A utility function for adding Model_Data.
#
# Parameters:
#   - data_a: first Model_data argument.
#   - data_b: second Model_data argument.
# Returns:
#   Model_Data with sum of all fields.
def add_data(data_a: TModel_Data, data_b: TModel_Data) -> TModel_Data:
  return add_data_weighted(data_a, data_b, 1, 1)

# A utility function for subtracting Model_Data.
#
# Parameters:
#   - data_a: first Model_data argument.
#   - data_b: second Model_data argument.
# Returns:
#   Model_Data with all fields from data_b subtracted from data_a.
def sub_data(data_a: TModel_Data, data_b: TModel_Data) -> TModel_Data:
  return add_data_weighted(data_a, data_b, 1, -1)

# A utility function for adding Model_Data with weights.
# Defaults to data_a's population.
#
# Parameters:
#   - data_a: first Model_data argument.
#   - data_b: second Model_data argument.
#   - weight_a: weight for first argument.
#   - weight_b: weight for second argument.
# Returns:
#   Model_Data with weighted sum of all fields, where the data is multiplied
#   with their respective weights before adding.
def add_data_weighted(data_a: TModel_Data, data_b: TModel_Data,
  weight_a: float, weight_b: float) -> TModel_Data:

  sum = cast(TModel_Data, {})
  for field in data_a.keys():
    if field == 'population':
      sum[field] = data_a[field]
    else:
      sum[field] = weight_a * data_a[field] + weight_b * data_b[field]
  return sum

# A utility function for multiplying Model_Data, with either
# another Model_Data where each field is multiplied, or a scalar
# which is multiplied with each field.
#
# Parameters:
#   - data: the data to be multiplied.
#   - factor: the factor to multiply by.
# Returns:
#   Model_Data with multiplied fields.
def mul_data(data: TModel_Data, factor: Union[TModel_Data, float]) -> TModel_Data:
  product = cast(TModel_Data, {})
  for field in data.keys():
    if field == 'population':
      product[field] = data[field]
    elif isinstance(factor, float) or isinstance(factor, int):
      product[field] = data[field] * factor
    else:
      product[field] = data[field] * factor[field]
  return product


# Normalizes the given data timeseries based on the population.
# All field values are divided by the population, including the population.
#
# Parameters:
#   - data_ts: timeseries of data to normalize.
# Returns:
#   Timeseries of normalized data.
def normalize(data_ts: Dict[date, TModel_Data]) -> Dict[date, TModel_Data]:
  # Iterate over each data point, normalizing by population
  normalized_ts = {}
  for day, data in data_ts.items():
    normalized_ts[day] = cast(TModel_Data, {})
    for field, val in data.items():
      if isinstance(val, float) or isinstance(val, int):
        normalized_ts[day][field] = val / data['population']
      else:
        normalized_ts[day][field] = val
  return normalized_ts


#################### MODELS FOR EPIDEMIC SPREAD ####################

# Type variable for epidemic_model for type checking
TEpidemic_Model = TypeVar('TEpidemic_Model', bound='Epidemic_Model')

# Abstract class for different epidemic models with their own unique
# sets of parameters for characterizing the epidemic.
#
# The Epidemic_Model class uses a generic TModel_Data type that corresponds
# with the model, which must be a subclass of Model_Data.
class Epidemic_Model(ABC, Generic[TModel_Data]):
  # Construct an epidemic model class given a tuple of the parameters:
  #
  # Parameters:
  #   - params: should be a tuple of the required parameters for the subclass.
  # Returns:
  #   A constructed object given a tuple of params.
  @classmethod
  @abstractmethod
  def init(cls: Type[TEpidemic_Model], params: Tuple) -> TEpidemic_Model:
    pass

  # Retrieves the datatype corresponding to the model
  #
  # Returns:
  #   The datatype that the model works with.
  @classmethod
  @abstractmethod
  def get_datatype(cls) -> Type[TModel_Data]:
    pass

  # Calculates the discrete timestep from the given data using the model.
  #
  # Parameters:
  #   - data: TModel_Data that represents the current state of the model.
  #
  # Returns:
  #   A TModel_Data object where each field is a discrete derivative, calculated
  #   from the current state of the epidemic.
  @abstractmethod
  def get_timestep(self, data: TModel_Data) -> TModel_Data:
    pass

  # Updates the model parameters by correcting towards the given data and delta,
  # incrementing by a factor of the training_weight.
  #
  # Should maintain all parameter invariants.
  #
  # Parameters:
  #   - data: TModel_Data that represents the current state of the model.
  #   - delta: TModel_data that contains the discrete derivatives calculated real-world data.
  #   - training_weight: value from 0-1 representing how much to correct the parameters
  #       to the expected parameters derived from delta.
  @abstractmethod
  def update_params(self, data: TModel_Data, delta: TModel_Data, training_weight: float):
    pass  

  # Trains the model by fitting the parameters to the given data.
  #
  # Parameters:
  #   - data_ts: a timeseries of the measured data, ordered from start date to end date.
  #   - start_date: the start date of data to anaylze.
  #   - end_date: the end date of data to anaylze.
  #   - data_weight: weight to give the data, in the range 0 to 1 inclusive.
  #   - learn_rate: scaling factor for each training step.
  #   - max_steps: the maximum number of training steps.
  #   - tolerance: min cost threshold to end training.
  # Returns:
  #   A list of TModel_Data representing the cost for each data field at each training iteration.
  @abstractmethod
  def train_model(self, data_ts: Dict[date, TModel_Data], start_date: date, end_date: date,
    data_weight: float, learn_rate: float, max_steps: int, tolerance: float) -> List[TModel_Data]:
    pass

  # Gets the next state from the current state and delta.
  #
  # Maintains invariant that all model state field values are greater or equal to 0.
  #
  # Parameters:
  #   - state: current state for model.
  #   - delta: change in state to add to current state.
  # Returns:
  #   - Model_Data with sum of fields, where all values are at least 0.
  def get_next_state(self, state: TModel_Data, delta: TModel_Data) -> TModel_Data:
    sum = add_data(state, delta)
    for field, val in sum.items():
      if isinstance(val, float):
        sum[field] = max(val, 0)
    return sum

  # Retrieves model predictions in timeseries format.
  #
  # Parameters:
  #   data_init: initial model state as a TModel_Data.
  #   start_date: date object for model start.
  #   end_date: date object for model end.
  # Returns:
  #   A dictionary mapping from the date to the TModel_Data of the predictions.
  def get_timeseries(self, data_init: TModel_Data, start_date: date, end_date: date) \
    -> Dict[date, TModel_Data]:
    
    # Initialize model state and timeseries return dict
    model_ts = {}
    model_state = data_init
    model_ts[start_date] = data_init

    # Iterate through dates and update/correct
    curr_date = start_date + timedelta(days=1)
    while curr_date <= end_date:
      # Update state using timestep
      delta = self.get_timestep(model_state)
      model_state = self.get_next_state(model_state, delta)
      model_ts[curr_date] = model_state

      curr_date = curr_date + timedelta(days=1)

    return model_ts

  # Calculates timeseries corrected between data and model, with additional forecast.
  #
  # Between start_date and end_date, returns corrected prediction between model and data
  # while incrementally updating its own model parameters to better fit the data.
  #
  # If there is no data for start_date in data_ts, the function will attempt to find the
  # first date with data before end_date to start with.
  #
  # If data is not present for a date between start_date and end_date, the model
  # will update its state based solely on its own prediction for that date.
  #
  # From forecast_len days past end_date, provides model predictions using corrected parameterrs.
  #
  # Parameters:
  #   data_ts: dict mapping from date to TModel_Data from real-world data.
  #   start_date: date object for model start.
  #   end_date: date object for model end.
  #   forecast_len: how many days past end_date to forecast using the model.
  #   data_weight: how much the data is favored v.s. the model, from 0 to 1.
  #   training_weight: how quickly to update data to measured data, from 0 to 1.
  # Returns:
  #   A dictionary mapping from the date to the TModel_Data of the corrected/forecasted data.
  # Raises:
  #   KeyError if data_ts does not contain data before end_date.
  def get_corrected_timeseries(self, data_ts: Dict[date, TModel_Data],
    start_date: date, end_date: date, forecast_len: int,
    data_weight: float, training_weight: float) -> Dict[date, TModel_Data]:

    # Initialize model state and timeseries return dict
    model_ts = {}
    curr_date = start_date
    while True:
      try:
        model_state = data_ts[curr_date]
        break
      except KeyError:
        curr_date += timedelta(days=1)
        if curr_date > end_date:
          raise KeyError("No data found before end date {}".format(end_date))
    model_ts[curr_date] = model_state
    
    # Corrective updates from start_date to end_date
    curr_date = curr_date + timedelta(days=1)
    while curr_date <= end_date:
      model_delta = self.get_timestep(model_state)
      try:
        # If data exists for date, find average delta
        data_delta = sub_data(data_ts[curr_date], model_state)
        avg_delta = add_data_weighted(data_delta, model_delta, data_weight, 1-data_weight)

        # Update parameters
        self.update_params(model_state, avg_delta, training_weight)
      except KeyError:
        # No data, update state solely on model
        avg_delta = model_delta

      # Update state and store results.
      model_state = self.get_next_state(model_state, avg_delta)
      model_ts[curr_date] = model_state

      curr_date = curr_date + timedelta(days=1)

    # Forecast forecast_len days from end_date 
    forecast = self.get_timeseries(model_state, end_date, end_date+timedelta(days=forecast_len))
    for dt in forecast.keys():
      model_ts[dt] = forecast[dt]

    return model_ts

# Basic SIR model, considering susceptible, infected, & recovered populations.
# Uses parameters beta to characterized the spread of a single case, and gamma
# to characterize how long the epidemic persists for a single case.
class SIR_Model(Epidemic_Model[SIR_Data]):
  # Invariants:
  #   - beta: beta >= 0
  #   - gamma: gamma >= 0
  beta: float   # Transmission parameter
  gamma: float  # Recovery rate

  # Constructs a SIR_Model object from the given parameters.
  #
  # Parameters:
  #   - beta: transmission parameter, average rate that infected infects susceptible
  #   - gamma: recovery rate, inverse of recovery time/infectious period
  def __init__(self, beta: float, gamma: float):
    self.beta = beta
    self.gamma = gamma

  @classmethod
  def init(cls, params: Tuple) -> SIR_Model:
    return SIR_Model(params[0], params[1])

  @classmethod
  def get_datatype(cls) -> Type[SIR_Data]:
      return SIR_Data

  def get_timestep(self, data: SIR_Data) -> SIR_Data:
    ds = -1 * self.beta * data['infected'] * data['susceptible'] / data['population']
    di = self.beta * data['infected'] * data['susceptible'] / data['population'] \
      - self.gamma * data['infected']
    dr = self.gamma * data['infected']
    return {'susceptible': ds, 'infected': di, 'recovered': dr, 'population': data['population']}

  def update_params(self, data: SIR_Data, delta: SIR_Data, training_weight: float):
    if not data['infected'] or not data['susceptible']:
      # skip for 0 values
      return

    # Parameters must be non-negative
    data_beta = max(-1 * delta['susceptible'] * data['population'] / (data['infected'] * data['susceptible']), 0)
    data_gamma = max(delta['recovered'] / data['infected'] , 0)

    self.beta = (1-training_weight) * self.beta + training_weight * data_beta
    self.gamma = (1-training_weight) * self.gamma + training_weight * data_gamma

  def train_model(self, data_ts: Dict[date, SIR_Data], start_date: date, end_date: date,
    data_weight: float,learn_rate: float, max_steps: int, tolerance: float) -> List[SIR_Data]:
    
    # Calculate the scale for each learning step
    scale = learn_rate*data_weight

    # Get normalized data
    norm_ts = normalize(data_ts)

    # Iterate over traning steps
    costs = []
    for _ in range(max_steps):
      # Compute parameter gradients and cost
      model_grad = self.gradient(norm_ts, start_date, end_date)
      costs.append(self.cost(norm_ts, start_date, end_date))

      # Scale gradients
      beta_diff = scale * model_grad.beta
      gamma_diff = scale * model_grad.gamma

      if np.abs(beta_diff) < tolerance and np.abs(gamma_diff) < tolerance:
        # Both parameters are within tolerance, finished
        break 
      if np.abs(beta_diff) >= tolerance:
        # Beta is not within tolerance, update with scaled gradient
        self.beta = max(0, self.beta - beta_diff)
      if np.abs(gamma_diff) >= tolerance:
        # Gamma is not within tolerance, update with scaled gradient
        self.gamma = max(0, self.gamma - gamma_diff)
    return costs
    
  # Retrieves the gradient for parameters, using a half mean squared error cost function
  #
  # Parameters:
  #   - data_ts: a timeseries of the measured data, ordered from start date to end date.
  #   - start_date: the start date of data to anaylze.
  #   - end_date: the end date of data to anaylze.
  # Returns:
  #   The gradient for each parameter in the fields of an SIR_Model object.
  def gradient(self, data_ts: Dict[date, SIR_Data], start_date: date, end_date: date) -> SIR_Model:
    # Weight for cost depending on day (later dates have more weight)
    total_weight = 0
    weight = 1

    # Sum component for each partial differentiation of SIR equations
    s_beta_sum = 0
    i_beta_sum = 0
    i_gamma_sum = 0
    r_gamma_sum = 0

    # Iterate through dates and accumulate cost sum.
    curr_date = start_date
    while curr_date < end_date:
      # Get SIR data
      sir = data_ts[curr_date]
      next_sir = data_ts[curr_date + timedelta(days=1)]
      model_sir = add_data(sir, self.get_timestep(data_ts[curr_date]))
      diff = sub_data(model_sir, next_sir)
      
      # Add sum component for cost derivatives, based on model functions
      s_beta_sum += weight * (-sir['infected']*sir['susceptible']/sir['population']) \
        * diff['susceptible']   # d(s_cost)/d(beta)
      i_beta_sum += weight * (sir['infected']*sir['susceptible']/sir['population']) \
        * diff['infected']      # d(i_cost)/d(beta)
      i_gamma_sum += weight * -sir['infected'] \
        * diff['infected']      # d(i_cost)/d(gamma)
      r_gamma_sum += weight * sir['infected'] \
        * diff['recovered']     # d(r_cost)/d(gamma)

      # Increment weight for later dates
      total_weight += weight
      weight += 1
      curr_date += timedelta(days=1)

    # Scale sum by factors in cost derivation
    if total_weight:
      s_beta_sum *= 1 / total_weight
      i_beta_sum *= 1 / total_weight
      i_gamma_sum *= 1 / total_weight
      r_gamma_sum *= 1 / total_weight

    # Calculate averages for beta/gamma
    beta_grad = (s_beta_sum + i_beta_sum) / 2
    gamma_grad = (i_gamma_sum + r_gamma_sum) / 2
    return SIR_Model(beta_grad, gamma_grad)

  # Retrieves the cost for parameters, using a half mean squared error cost function
  #
  # Parameters:
  #   - data_ts: a timeseries of the measured data, ordered from start date to end date.
  #   - start_date: the start date of data to anaylze.
  #   - end_date: the end date of data to anaylze.
  # Returns:
  #   The cost for each data field stored in an SIR_Data object
  def cost(self, data_ts: Dict[date, SIR_Data], start_date: date, end_date: date) -> SIR_Data:
    # Weight for cost depending on day (later dates have more weight)
    total_weight = 0
    weight = 1
    sum: SIR_Data = {'susceptible': 0, 'infected': 0, 'recovered': 0, 'population': 0}

    # Iterate through dates and accumulate cost sum.
    curr_date = start_date
    while curr_date < end_date:
      sir = data_ts[curr_date]
      next_sir = data_ts[curr_date + timedelta(days=1)]
      model_sir = add_data(sir, self.get_timestep(data_ts[curr_date]))
      diff = sub_data(model_sir, next_sir)

      # Add sum component for cost derivatives, based on model functions
      sum = add_data_weighted(sum, mul_data(diff, diff), 1, weight)  

      # Increment weight for later dates
      total_weight += weight
      weight += 1
      curr_date += timedelta(days=1)

    # Scale sum by factors in cost derivation
    if total_weight:
      mul_data(sum, 1 / total_weight)
    return sum

# SIRD model, considering susceptible, infected, recovered, and dead populations.
class SIRD_Model(Epidemic_Model[SIRD_Data]):
  def __init__(self):
    # TODO: update with SIRD parameters
    pass

  @classmethod
  def get_datatype(cls) -> Type[SIRD_Data]:
    return SIRD_Data

  def get_timestep(self, data: SIRD_Data) -> SIRD_Data:
    # TODO: update with SIRD equations
    return NotImplemented

  def update_params(self, data: SIRD_Data, delta: SIRD_Data, training_weight: float):
    # TODO: update with SIRD equations
    pass
