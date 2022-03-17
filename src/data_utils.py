# This file contains utility classes for interfacing with epidemic data.

import csv
import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from urllib.request import urlopen
from datetime import date, timedelta
from os.path import exists

from models import SIR_Data, TModel_Data, add_data, mul_data

# For John Hopkins University data repository (https://github.com/CSSEGISandData/COVID-19)
JHU_FIRST_DAY = date(2020, 5, 29)
JHU_END_DAY = date.today() - timedelta(days = 1)
JHU_COUNTRY_DATAPATH = "./data/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/"
JHU_DATE_FORMAT = "%m-%d-%Y"
JHU_JSON_DATAPATH = "./data/JHU_Country_Timeseries.json"

# For CAN, which stands for Covid Act Now (https://apidocs.covidactnow.org/)
CAN_API_KEY = "3e31ceb6829a4908a92a57d5be127edc"
CAN_US_COUNTY_URL = "https://api.covidactnow.org/v2/county/{}.timeseries.json?apiKey="+CAN_API_KEY
CAN_US_STATE_URL = "https://api.covidactnow.org/v2/states.timeseries.json?apiKey="+CAN_API_KEY
CAN_DATE_FORMAT = "%Y-%m-%d"
CAN_STATE_JSON_DATAPATH = "./data/CAN_State_Timeseries.json"
CAN_COUNTY_JSON_DATAPATH = "./data/county/CAN_{}_County_Timeseries.json"

# For OWID, which stands for Our World In Data (https://github.com/owid/covid-19-data/tree/master/public/data)
OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.json"
OWID_DATE_FORMAT = "%Y-%m-%d"
OWID_JSON_DATAPATH = "./data/OWID_Country_Timeseries.json"

# Full Width at Half Maximum for Gaussian Smoothing
SIGMA = 2.0

# Abstract class for epidemic data.
class Epidemic_Data(ABC):
    # Retrieves list of all locations.
    #
    # Returns:
    #   Set of locations associated with data.
    @abstractmethod
    def get_locations(self) -> List[str]:
        pass

    # Retrieves population for given location. Raises KeyError if not found.
    #
    # Parameters:
    #   - loc: location to retrieve population for.
    # Returns:
    #   Population count for given location.
    # Raises:
    #   KeyError if data not found for location.
    @abstractmethod
    def get_population(self, loc: str) -> int:
        pass

    # Retrieves cases for given location and date. Tries all previous dates,
    # raises KeyError if no data found.
    #
    # Parameters:
    #   - loc: location to retrieve cases for.
    #   - date: date to retrieve cases for.
    # Returns:
    #   Case count for given location and date.
    # Raises:
    #   KeyError if data not found for location and date.
    @abstractmethod
    def get_cases(self, loc: str, date: date) -> int:
        pass

    # Retrieves vaccinations for given location and date. Tries all previous dates,
    # raises KeyError if no data found.
    #
    # Parameters:
    #   - loc: location to retrieve vaccinations for.
    #   - date: date to retrieve vaccinations for.
    # Returns:
    #   Vaccination count for given location and date.
    # Raises:
    #   KeyError if data not found for location and date.
    @abstractmethod
    def get_vaccinations(self, loc: str, date: date) -> int:
        pass

    # Returns SIR data for the given date, location, and infectious_period of disease.
    #
    # Parameters:
    #   loc: location to retrieve data for.
    #   curr_date: date to retrieve data for.
    #   infectious_period: how long the disease is infectious for, on average.
    # Returns:
    #   SIR_Data containing susceptible, infected, and recovered data for date.
    # Raises:
    #   KeyError if data not found for location and date.
    def get_SIR_for_day(self, loc: str, curr_date: date, infectious_period: int) -> SIR_Data:
        prev_date = curr_date - timedelta(days=infectious_period)

        population = self.get_population(loc)
        curr_cases = self.get_cases(loc, curr_date)
        prev_cases = self.get_cases(loc, prev_date)
        vaccinations = self.get_vaccinations(loc, curr_date)

        r = prev_cases
        i = curr_cases - r
        s = max(0, population - vaccinations - (r + i))     
        return {'susceptible': s, 'infected': i, 'recovered': r, 'population': population}

    # Returns data as a timeseries. Skips over dates/locations with no data.
    #
    # Parameters:
    #   loc: location to retrieve data for.
    #   start_date: start date of timeseries.
    #   end_date: end date of timeseries.
    #   infectious_period: how long the disease is infectious for, on average.
    #   datatype: type of data to retrieve data for.
    #       Currently supports SIR_Data.
    # Return:
    #   Dict mapping date to Model_Data object, with type specified by datatype.
    def get_timeseries(self, loc: str, start_date: date, end_date: date,
        infectious_period: int, datatype: Type[TModel_Data]) -> Dict[date, TModel_Data]:

        if datatype == SIR_Data:
            get_data_for_day = self.get_SIR_for_day
        else:
            return NotImplemented

        while start_date <= end_date:
            try:
                get_data_for_day(loc, start_date, infectious_period)
                break
            except KeyError:
                start_date = start_date + timedelta(days=1)

        data_ts = {}
        curr_date = start_date
        while curr_date <= end_date:
            try:
                data_ts[curr_date] = get_data_for_day(loc, curr_date, infectious_period)
            except KeyError:
                # no more data available
                break
            curr_date = curr_date + timedelta(days=1)

        smoothed_ts = {}
        dates = np.array(sorted(data_ts.keys()))
        values = list(data_ts[d] for d in dates)
        for date in dates:
            kernel = np.exp(
                np.array(list(-d.days**2/((2 * (SIGMA**2))) for d in (dates - date))).astype(float)
            )
            kernel /= kernel.sum()

            sum = mul_data(values[0], kernel[0])
            for i in range(1, len(dates)):
                sum = add_data(sum, mul_data(values[i], kernel[i]))
            smoothed_ts[date] = sum

        return smoothed_ts

# Epidemic data for COVID-19 retrieved from the John Hopkin's University GitHub repository.
class JHU_COVID_Data(Epidemic_Data):
    # Maps country name to dictionary containing data.
    data: Dict[str, Dict[str, Any]]

    # Constructs a JHU_COVID_Data object by parsing through CSV files in submodule for
    # JHU GitHub. Caches result in json, and tries to reitreve data directly.
    #
    # Parameters:
    #   - update: whether or not to update data from JHU GitHub.
    def __init__(self, update: bool = False):
        if update:
            # Update submodule and remove outdated JSON file.
            os.system("git submodule update --remote")
            os.system("rm {}".format(JHU_JSON_DATAPATH))

        try:
            # Try to open cached data from json
            json_datafile = open(JHU_JSON_DATAPATH, 'r')
            self.data = json.load(json_datafile)
            json_datafile.close()
        except FileNotFoundError:
            print("No JSON found, parsing country data...")
            self.data = {}
            curr_date = JHU_FIRST_DAY
            while curr_date <= JHU_END_DAY:
                date_filename = curr_date.strftime(JHU_DATE_FORMAT + ".csv")
                curr_date_populations = {}
                running_avg_incidence = 0.0
                row_count = 0
                if exists(JHU_COUNTRY_DATAPATH + date_filename):
                    with open(JHU_COUNTRY_DATAPATH + date_filename, 'r') as csvfile:
                        csvreader = csv.reader(csvfile)
                        next(csvreader)
                        for row in csvreader:
                            if row[3] not in self.data:
                                self.data[row[3]] = {
                                    'Population': 0,
                                    'Timeseries': {}
                                }
                            if curr_date.strftime(JHU_DATE_FORMAT) not in self.data[row[3]]['Timeseries']:
                                self.data[row[3]]['Timeseries'][curr_date.strftime(JHU_DATE_FORMAT)] = {
                                    'Cases': 0
                                }
                            self.data[row[3]]['Timeseries'][curr_date.strftime(JHU_DATE_FORMAT)]['Cases'] += int(row[7])
                            incidence_rate = running_avg_incidence
                            if row[12] != '':
                                incidence_rate = float(row[12])
                                running_avg_incidence = (running_avg_incidence * row_count + incidence_rate) / (row_count + 1)
                                row_count += 1
                            additional_population = 0
                            if incidence_rate != 0.0:
                                additional_population = int(row[7]) / incidence_rate * 100000
                            if row[3] not in curr_date_populations:
                                curr_date_populations[row[3]] = 0
                            curr_date_populations[row[3]] += additional_population
                    for loc in curr_date_populations:
                        self.data[loc]['Population'] = max(self.data[loc]['Population'], curr_date_populations[loc])
                curr_date = curr_date + timedelta(days = 1)
            with open(JHU_JSON_DATAPATH, 'w') as fp:
                # Cache data as json
                json.dump(self.data, fp)

    def get_locations(self) -> List[str]:
        return list(self.data.keys())

    def get_population(self, loc: str) -> int:
        if not loc in self.data or self.data[loc] is None:
            raise KeyError("No data for country {}".format(loc))
            
        population = self.data[loc]['Population']
        if population is None:
            raise KeyError("Population not found for couuntry {}".format(loc))
        return int(population)

    def get_cases(self, loc: str, date: date) -> int:
        if not loc in self.data or self.data[loc] is None:
            raise KeyError("No data for country {}".format(loc))
        if not date.strftime(JHU_DATE_FORMAT) in self.data[loc]["Timeseries"]:
            raise KeyError("No data for country {} on date {}".format(loc, date.strftime(JHU_DATE_FORMAT)))

        while True:
            try:
                cases = self.data[loc]["Timeseries"][date.strftime(JHU_DATE_FORMAT)]["Cases"]
                if not cases is None:
                    return int(cases)
                date -= timedelta(days=1)
            except KeyError:
                raise KeyError("Cases not found for any date prior to {} in country {}".format(str(date), loc))

    def get_vaccinations(self, loc: str, date: date) -> int:
        return 0    # TODO: Get actual vaccination data for countries

# Epidemic data for COVID-19 retrieved from the Covid Act Now API.
class CAN_COVID_Data(Epidemic_Data):
    # Data organized by date and state for COVID-19 in the US
    #   The data field is a dict:
    #       key: location, either state, country, or county
    #           for state, must be 2-letter ANSI code
    #           for country, must be 2-letter ISO-3166 code
    #           for county, must be county name
    #       value: {
    #           'loc_data': location json data,
    #           'metrics': metricsTimeseries,
    #           'actuals': actualsTimeseries,
    #           'riskLevels': riskLevelsTimeseries,
    #           'cdcTransmissionLevel': cdcTransmissionLevelTimeseries
    #       }
    #   For the metrics, actuals, riskLevels, and cdcTransmissionLevel timeseries dicts:
    #       key: date formatted by "yyyy-mm-dd"
    #       value: corresponding json data
    #   For json data format, see:
    #       https://apidocs.covidactnow.org/api#tag/State-Data/paths/~1states.timeseries.json?apiKey={apiKey}/get
    data: Dict[str, Dict[str, Any]]

    # Initializes data object by parsing Covid Act Now JSON and updating data if necessary.
    #
    # Parameters:
    #   - loc_type: must be either "state" or "county".
    #   - state: state for counties, if loc_type is "county".
    #   - update: whether or not to update.
    def __init__(self, loc_type: str, state: str | None = None, update: bool = True): 
        self.loc_type = loc_type
        if loc_type == 'county':
            datapath = CAN_COUNTY_JSON_DATAPATH.format(state)
            url = CAN_US_COUNTY_URL.format(state)
        else:
            datapath = CAN_STATE_JSON_DATAPATH
            url = CAN_US_STATE_URL

        if update and os.path.exists(datapath):
            os.remove(datapath)
    
        try:
            # Try to open cached data from json
            json_datafile = open(datapath, 'r')
            self.data = json.load(json_datafile)
            json_datafile.close()
        except FileNotFoundError:
            print("Retreiving data...")
            with urlopen(url) as url_data:
                loc_array = json.loads(url_data.read().decode())
            
            # Convert JSON object into dict
            print("Indexing data...")
            self.data = {}
            for loc_data in loc_array:
                # Convert timeseries data to dict, and erase data from json object
                metricsTimeseriesDict = {}
                for metrics in loc_data['metricsTimeseries']:
                    date_obj = metrics['date']
                    metricsTimeseriesDict[date_obj] = metrics
                loc_data['metricsTimeseries'] = None

                actualsTimeseriesDict = {}
                for actuals in loc_data['actualsTimeseries']:
                    date_obj = actuals['date']
                    actualsTimeseriesDict[date_obj] = actuals
                loc_data['actualsTimeseries'] = None

                riskLevelsTimeseriesDict = {}
                for riskLevels in loc_data['riskLevelsTimeseries']:
                    date_obj = riskLevels['date']
                    riskLevelsTimeseriesDict[date_obj] = riskLevels
                loc_data['riskLevelsTimeseries'] = None

                cdcTransmissionLevelTimeseriesDict = {}
                for cdcTransmissionLevel in loc_data['cdcTransmissionLevelTimeseries']:
                    date_obj = cdcTransmissionLevel['date']
                    cdcTransmissionLevelTimeseriesDict[date_obj] = cdcTransmissionLevel
                loc_data['cdcTransmissionLevelTimeseries'] = None

                self.data[loc_data[loc_type]] = {
                    'loc_data': loc_data,
                    'metrics': metricsTimeseriesDict,
                    'actuals': actualsTimeseriesDict,
                    'riskLevels': riskLevelsTimeseriesDict,
                    'cdcTransmissionLevel': cdcTransmissionLevelTimeseriesDict
                }

            with open(datapath, 'w') as fp:
                json.dump(self.data, fp)

    def get_locations(self) -> List[str]:
        return list(self.data.keys())

    def get_population(self, loc: str) -> int:
        if not loc in self.data or self.data[loc] is None:
            raise KeyError("No data for {} {}".format(self.loc_type, loc))

        population = self.data[loc]['loc_data']['population']
        if population is None:
            raise KeyError("Population not found for {} {}".format(self.loc_type, loc))
        return int(population)

    def get_cases(self, loc: str, date: date) -> int:
        data = self.data[loc]['actuals'][date.strftime(CAN_DATE_FORMAT)]
        if 'cases' not in data or data['cases'] is None:
            try:
                data['cases'] = self.get_cases(loc, date-timedelta(days=1))
            except KeyError:
                data['cases'] = -1
        if data['cases'] == -1:
            raise KeyError("Cases not found for any date prior to {} in {} {}".format(str(date), self.loc_type, loc))
        return int(data['cases'])

    def get_vaccinations(self, loc: str, date: date) -> int:
        data = self.data[loc]['actuals'][date.strftime(CAN_DATE_FORMAT)]
        if 'vaccinationsCompleted' not in data or data['vaccinationsCompleted'] is None:
            try:
                data['vaccinationsCompleted'] = self.get_vaccinations(loc, date-timedelta(days=1))
            except KeyError:
                data['vaccinationsCompleted'] = -1
        if data['vaccinationsCompleted'] == -1:
            raise KeyError("Vaccinations not found for any date prior to {} in {} {}".format(str(date), self.loc_type, loc))
        return int(data['vaccinationsCompleted'])


# Epidemic data for COVID-19 retrieved from the Our World In Data API.
class OWID_COVID_Data(Epidemic_Data):
    # Data organized by date and country for COVID-19
    #   The data field is a dict:
    #       key: country name,
    #       value: {
    #           'loc_data': location json data,
    #           'ts': data timeseries
    #       }
    #   For the data timeseries dict:
    #       key: date formatted by "yyyy-mm-dd"
    #       value: corresponding json data
    #   For json data format, see:
    #       https://covid.ourworldindata.org/data/owid-covid-data.json
    data: Dict[str, Dict[str, Any]]  # data timeseries

    # Initializes data object by parsing Our World In Data JSON and updating data if necessary.
    def __init__(self, update: bool = True):
        if update and os.path.exists(OWID_JSON_DATAPATH):
            os.remove(OWID_JSON_DATAPATH)

        try:
            # Try to open cached data from json
            json_datafile = open(OWID_JSON_DATAPATH, 'r')
            self.data = json.load(json_datafile)
            json_datafile.close()
        except FileNotFoundError:
            print("Retreiving data...")
            with urlopen(OWID_URL) as url_data:
                loc_dict = json.loads(url_data.read().decode())

            # Convert JSON object into dict
            print("Indexing data...")
            self.data = {}
            for loc_ISO_code in loc_dict:
                # Convert timeseries data to dict, and erase data from json object
                timeseriesDict = {}
                for metrics in loc_dict[loc_ISO_code]['data']:
                    date_obj = metrics['date']
                    timeseriesDict[date_obj] = metrics
                loc_dict[loc_ISO_code]['data'] = None

                self.data[loc_dict[loc_ISO_code]['location']] = {
                    'loc_data': loc_dict[loc_ISO_code],
                    'ts': timeseriesDict
                }
            with open(OWID_JSON_DATAPATH, 'w') as fp:
                json.dump(self.data, fp)

    def get_locations(self) -> List[str]:
        return list(self.data.keys())

    def get_population(self, loc: str) -> int:
        if not loc in self.data or self.data[loc] is None:
            raise KeyError("No data for country {}".format(loc))

        population = self.data[loc]['loc_data']['population']
        if population is None:
            raise KeyError("Population not found for country {}".format(loc))
        return int(population)

    def get_cases(self, loc: str, date: date) -> int:
        data = self.data[loc]['ts'][date.strftime(OWID_DATE_FORMAT)]
        if 'total_cases' not in data or data['total_cases'] is None:
            try:
                data['total_cases'] = self.get_cases(loc, date-timedelta(days=1))
            except KeyError:
                data['total_cases'] = -1
        if data['total_cases'] == -1:
            raise KeyError("Cases not found for any date prior to {} in country {}".format(str(date), loc))
        return int(data['total_cases'])

    def get_vaccinations(self, loc: str, date: date) -> int:
        data = self.data[loc]['ts'][date.strftime(OWID_DATE_FORMAT)]
        if 'people_fully_vaccinated' not in data or data['people_fully_vaccinated'] is None:
            try:
                data['people_fully_vaccinated'] = self.get_vaccinations(loc, date-timedelta(days=1))
            except KeyError:
                data['people_fully_vaccinated'] = -1
        if data['people_fully_vaccinated'] == -1:
            raise KeyError("Vaccinations not found for any date prior to {} in country {}".format(str(date), loc))
        return int(data['people_fully_vaccinated'])
