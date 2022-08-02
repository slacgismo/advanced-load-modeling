# ALM model fuctions support

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
from dateutil import *
import scipy.stats as st
import control as ct

    
# Load data functions

def NA(x):
    try:
        return round(float(x),1)
    except:
        return float('NAN')

def read_power_file(power_file_csv, datetime_col_name, power_col_name, DatetimeFormat = '%m/%d/%y %H:%M'):
    """
    Load power data from CSV file
    
    Input parameters:
    
        'power_file_csv' (str)     input csv file name for power data
        
        'datetime_col_name'(str)   input column name for datetime data
        
        'power_col_name' (str)     input column name for power data
                  
        'DatetimeFormat' (str)     input datetime format (defult format "%m/%d/%y %H:%M")
         
    Returns
    
        Power DataFrame - hourly power data with datetime index
        
    The user imports the specified CSV file and specify power and datetime 
    column names and format, then returns the result as a DataFrame.       
    """
    def timestamp(x):
        return datetime.strptime(x,DatetimeFormat)
    feeder_power = pd.read_csv('feeder_power.csv', converters = {datetime_col_name: timestamp, power_col_name: NA},
                                usecols = [datetime_col_name,power_col_name])    
    # If data is subhourly, the hourly average power will be computed
    index = list(map(lambda t:datetime(t.year,t.month,t.day,t.hour),pd.DatetimeIndex(feeder_power[datetime_col_name])))
    feeder_power = feeder_power.groupby(index).feeder_power.mean()
    feeder_power = pd.DataFrame(feeder_power, index = feeder_power.index)
    feeder_power.columns = ["power"]
    feeder_power.index.name = "datetime"
    
    return feeder_power.round(1)

def read_weatherfile(weather_file_csv, datetime_column_name, temperature_column_name, solar_column_name = 'none',
                    DatetimeFormat = '%m/%d/%y %H:%M'):
    """
    Load weather data from CSV file
    
    Input parameters:
    
        'weather_file_csv' (str)       input csv file name for weather data
        
        'datetime_col_name' (str)      input column name for datetime data                                   
        
        'temperature_col_name' (str)   input column name for temperature/heat_index data (degF)
        
        'solar_col_name' (str)         input column name for solar index data (W/defF)
        
        'DatetimeFormat' (str)       input datetime format (defult format "%m/%d/%y %H:%M")
         
    Returns
    
        Weather DataFrame - Temperature and Solar gain data with datetime index
        
    The user imports the specified CSV file and specify temperature and optional
    solar gain column names and format, then return the result as a DataFrame. 
    """     
    def timestamp(x):
        return datetime.strptime(x,DatetimeFormat)    
    if solar_column_name == 'none':        
        feeder_weather = pd.read_csv(weather_file_csv, converters = { datetime_column_name: timestamp,
                                                                         temperature_column_name: NA},                                                                  usecols = [datetime_column_name,temperature_column_name],
                                    keep_default_na = False,
                                    index_col = [datetime_column_name])
    else:
        feeder_weather = pd.read_csv(weather_file_csv, converters = { datetime_column_name: timestamp,
                                                                     temperature_column_name: NA, solar_column_name: NA},         
                                    usecols = [datetime_column_name,temperature_column_name, solar_column_name],
                                    keep_default_na = False,
                                    index_col = [datetime_column_name])    
    # If data is subhourly, the hourly average power will be computed
    index = list(map(lambda t:datetime(t.year,t.month,t.day,t.hour),feeder_weather.index))
    weather = feeder_weather.groupby(index).mean().dropna()    
    if solar_column_name == 'none':         
        weather.columns = ["temperature"]
    else:
        weather.columns = ["solar", "temperature"]        
    weather.index.name = "datetime"   
    
    return weather


# Static model functions   

def get_baseload(data,
        Tbase = 'auto',
        Tdiff = 10):
    
    """
    Get the base static load model
    
    Input parameters:
    
         data (DataFrame)          input dataframe including power and temperature load data
        
         Tbase (float or 'auto')   base temperature for static model, if 'auto' it finds the
                                   temperature that minimizes the slope at base region
                                   
         Tdiff (float)             temperature difference between Theat and Tcool (base region)
                                   (defult 10 degF)
         
    Returns
    
        slope_base (float)         slope loof base region 
        
        Tbase (float)              base temperature in degF 
        
        base (DataFrame)           base region dataframe (within +- 5degF from Tbase)
        
        intercept (float)          intercept value for base region static power equation      
     
    """   

    if Tbase == 'auto':
        slope_base=np.Infinity
        MaxSlope = 0.1
        Tmin = 0
        Tmax = 120
        Tbase = (Tmin + Tmax)/2
        n = 1
        MaxIterations = 100
        while abs(slope_base) > MaxSlope and n < MaxIterations:
            base = data[(data['temperature']-Tbase).abs()<=Tdiff/2]
            T = base['temperature']
            P = base['power']
            slope_base, intercept, rvalue, pvalue, stderr = st.linregress(T,P)
            load = intercept + slope_base*Tbase
            if slope_base > 0:
                Tmax = Tbase
                Tbase= (Tmin + Tmax)/2
            else:
                Tmin = Tbase
                Tbase= (Tmin + Tmax)/2               
            n += 1
    else:
        base = data[(data['temperature']-Tbase).abs()<=Tdiff/2]
        T = base['temperature']
        P = base['power']
        slope_base, intercept, rvalue, pvalue, stderr = st.linregress(T,P)
        load = intercept + slope_base*Tbase
        
    return slope_base, Tbase, base, intercept


def get_slopes(heating,cooling):
    
    """
    Get the heating and cooling regions for the static load model
    
    Input parameters:
    
        heating (DataFrame)    data set for heating region (must include power and temperature data)
        
        cooling (DataFrame)    data set for cooling region (must include power and temperature data)
         
    Returns
    
        heat_slope (float)     slope of heating region for static model (power per Temperature - MW/degF)
        
        cool_slope (float)     slope of cooling region for static model (power per Temperature - MW/degF)
        
        h0 (fooat)             power intercept at T = 0 for heating region equation
        
        c0 (float)             Intercept value at T = 0 for cooling region equation
        
    This fuction performs a lineaer regression on the data set to determine 
    the slopes and intercepts for the heating and cooling regions. 
    """ 
    
    if not heating.empty:
        T = np.array(heating['temperature'])
        P = np.array(heating['power'])
        heat_slope, h0, hr, hp, hs = st.linregress(T,P)
    else:
        heat_slope = None

    if not cooling.empty:
        T = np.array(cooling['temperature'])
        P = np.array(cooling['power'])
        cool_slope, c0, cr, cp, cs = st.linregress(T,P)
    else:
        cool_slope = None

    return heat_slope, cool_slope, h0, c0


# Dynamic and Hyprid model support functions

def feature_selection(dataframe, months_div, days_div, hours_div):
    """
    Function to specify features for month's, day's, and hour's divisor for a given dataframe with datetime index
    
    Input parameters:
    
        dataframe (DataFrame)           data set for feeder power, temperature, and optional solar with datetime index
        
        months_div (2,3,4,6,12)         months divisor in a year
        
        days_div (2,7)                  days divisor in a week
        
        hours_div (2,3,4,6,8,12,24)     hours divisor in a day
        
    Returns
    
        feeder_ind (DataFrame)          same input dataframe with added features' columns
        
    """    
    feeder_ind = dataframe.copy()
    
    # month divisor    
    if months_div == 12:
        feeder_ind['Jan'] = np.where(feeder_ind.index.month == 1,1,0)
        feeder_ind['Feb'] = np.where(feeder_ind.index.month == 2,1,0)
        feeder_ind['Mar'] = np.where(feeder_ind.index.month == 3,1,0)
        feeder_ind['Apr'] = np.where(feeder_ind.index.month == 4,1,0)
        feeder_ind['May'] = np.where(feeder_ind.index.month == 5,1,0)
        feeder_ind['Jun'] = np.where(feeder_ind.index.month == 6,1,0)
        feeder_ind['Jul'] = np.where(feeder_ind.index.month == 7,1,0)
        feeder_ind['Aug'] = np.where(feeder_ind.index.month == 8,1,0)
        feeder_ind['Sep'] = np.where(feeder_ind.index.month == 9,1,0)
        feeder_ind['Oct'] = np.where(feeder_ind.index.month == 10,1,0)
        feeder_ind['Nov'] = np.where(feeder_ind.index.month == 11,1,0)
        feeder_ind['Dec'] = np.where(feeder_ind.index.month == 12,1,0)
                
    elif months_div == 6:
        feeder_ind['Jan Feb'] = np.where((np.logical_or(feeder_ind.index.month==1,feeder_ind.index.month==2)),1,0)
        feeder_ind['Mar Apr'] = np.where((np.logical_or(feeder_ind.index.month==3,feeder_ind.index.month==4)),1,0)
        feeder_ind['May Jun'] = np.where((np.logical_or(feeder_ind.index.month==5,feeder_ind.index.month==6)),1,0)
        feeder_ind['Jul Aug'] = np.where((np.logical_or(feeder_ind.index.month==7,feeder_ind.index.month==8)),1,0)
        feeder_ind['Sep Oct'] = np.where((np.logical_or(feeder_ind.index.month==9,feeder_ind.index.month==10)),1,0)
        feeder_ind['Nov Dec'] = np.where((np.logical_or(feeder_ind.index.month==11,feeder_ind.index.month==12)),1,0)
        
    elif months_div == 4:
        feeder_ind['W'] = np.where((np.logical_or(np.logical_or(feeder_ind.index.month==12,
                                                                feeder_ind.index.month==1),
                                                                feeder_ind.index.month==2)),1,0)
        feeder_ind['Sp'] = np.where((np.logical_or(np.logical_or(feeder_ind.index.month==3,
                                                                 feeder_ind.index.month==4),
                                                                 feeder_ind.index.month==5)),1,0)
        feeder_ind['Su'] = np.where((np.logical_or(np.logical_or(feeder_ind.index.month==6,
                                                                 feeder_ind.index.month==7),
                                                                 feeder_ind.index.month==8)),1,0)
        feeder_ind['F'] = np.where((np.logical_or(np.logical_or(feeder_ind.index.month==9,
                                                                feeder_ind.index.month==10),
                                                                feeder_ind.index.month==11)),1,0)        
    elif months_div == 3:
        feeder_ind['fst']= np.where(np.logical_and(feeder_ind.index.month >= 1, feeder_ind.index.month <=4),1,0)
        feeder_ind['sec']= np.where(np.logical_and(feeder_ind.index.month >= 5, feeder_ind.index.month <=8),1,0)
        feeder_ind['thr']= np.where(np.logical_and(feeder_ind.index.month >= 9, feeder_ind.index.month <=12),1,0)
        
    elif months_div == 2:
        feeder_ind['first_half']= np.where(np.logical_and(feeder_ind.index.month >= 1, feeder_ind.index.month <=6),1,0)
        feeder_ind['sec_half']= np.where(np.logical_and(feeder_ind.index.month >= 7, feeder_ind.index.month <=12),1,0)

    # day type             
    if days_div == 2:
        feeder_ind['WD'] = np.where(feeder_ind.index.weekday < 5,1,0)
        feeder_ind['WE'] = np.where(feeder_ind.index.weekday >= 5,1,0)

    elif days_div == 7:
        feeder_ind['M'] = np.where(feeder_ind.index.weekday == 0,1,0)
        feeder_ind['Tu'] = np.where(feeder_ind.index.weekday == 1,1,0)
        feeder_ind['W'] = np.where(feeder_ind.index.weekday == 2,1,0)
        feeder_ind['Th'] = np.where(feeder_ind.index.weekday == 3,1,0)
        feeder_ind['F'] = np.where(feeder_ind.index.weekday == 4,1,0)
        feeder_ind['Sa'] = np.where(feeder_ind.index.weekday == 5,1,0)
        feeder_ind['Su'] = np.where(feeder_ind.index.weekday == 6,1,0)
           
    # hour type
    if hours_div == 24:
        feeder_ind['0'] = np.where(feeder_ind.index.hour == 0,1,0)
        feeder_ind['1'] = np.where(feeder_ind.index.hour == 1,1,0)
        feeder_ind['2'] = np.where(feeder_ind.index.hour == 2,1,0)
        feeder_ind['3'] = np.where(feeder_ind.index.hour == 3,1,0)
        feeder_ind['4'] = np.where(feeder_ind.index.hour == 4,1,0)
        feeder_ind['5'] = np.where(feeder_ind.index.hour == 5,1,0)
        feeder_ind['6'] = np.where(feeder_ind.index.hour == 6,1,0)
        feeder_ind['7'] = np.where(feeder_ind.index.hour == 7,1,0)
        feeder_ind['8'] = np.where(feeder_ind.index.hour == 8,1,0)
        feeder_ind['9'] = np.where(feeder_ind.index.hour == 9,1,0)
        feeder_ind['10'] = np.where(feeder_ind.index.hour == 10,1,0)
        feeder_ind['11'] = np.where(feeder_ind.index.hour == 11,1,0)
        feeder_ind['12'] = np.where(feeder_ind.index.hour == 12,1,0)
        feeder_ind['13'] = np.where(feeder_ind.index.hour == 13,1,0)
        feeder_ind['14'] = np.where(feeder_ind.index.hour == 14,1,0)
        feeder_ind['15'] = np.where(feeder_ind.index.hour == 15,1,0)
        feeder_ind['16'] = np.where(feeder_ind.index.hour == 16,1,0)
        feeder_ind['17'] = np.where(feeder_ind.index.hour == 17,1,0)
        feeder_ind['18'] = np.where(feeder_ind.index.hour == 18,1,0)
        feeder_ind['19'] = np.where(feeder_ind.index.hour == 19,1,0)
        feeder_ind['20'] = np.where(feeder_ind.index.hour == 20,1,0)
        feeder_ind['21'] = np.where(feeder_ind.index.hour == 21,1,0)
        feeder_ind['22'] = np.where(feeder_ind.index.hour == 22,1,0)
        feeder_ind['23'] = np.where(feeder_ind.index.hour == 23,1,0)

    elif hours_div == 12:
        feeder_ind['0-1'] = np.where(np.logical_or(feeder_ind.index.hour==0,feeder_ind.index.hour==1),1,0)
        feeder_ind['2-3'] = np.where(np.logical_or(feeder_ind.index.hour==2,feeder_ind.index.hour==3),1,0)
        feeder_ind['5-5'] = np.where(np.logical_or(feeder_ind.index.hour==4,feeder_ind.index.hour==5),1,0)
        feeder_ind['6-7'] = np.where(np.logical_or(feeder_ind.index.hour==6,feeder_ind.index.hour==7),1,0)
        feeder_ind['8-9'] = np.where(np.logical_or(feeder_ind.index.hour==8,feeder_ind.index.hour==9),1,0)
        feeder_ind['10-11'] = np.where(np.logical_or(feeder_ind.index.hour==10,feeder_ind.index.hour==11),1,0)
        feeder_ind['12-13'] = np.where(np.logical_or(feeder_ind.index.hour==12,feeder_ind.index.hour==13),1,0)
        feeder_ind['14-15'] = np.where(np.logical_or(feeder_ind.index.hour==14,feeder_ind.index.hour==15),1,0)
        feeder_ind['16-17'] = np.where(np.logical_or(feeder_ind.index.hour==16,feeder_ind.index.hour==17),1,0)
        feeder_ind['18-19'] = np.where(np.logical_or(feeder_ind.index.hour==18,feeder_ind.index.hour==19),1,0)
        feeder_ind['20-21'] = np.where(np.logical_or(feeder_ind.index.hour==20,feeder_ind.index.hour==21),1,0)
        feeder_ind['22-23'] = np.where(np.logical_or(feeder_ind.index.hour==22,feeder_ind.index.hour==23),1,0)
        
    elif hours_div == 8:
        feeder_ind['0-2'] = np.where(np.logical_and(feeder_ind.index.hour >= 0, feeder_ind.index.hour <=2),1,0)
        feeder_ind['3-5'] = np.where(np.logical_and(feeder_ind.index.hour >= 3, feeder_ind.index.hour <=5),1,0)
        feeder_ind['6-8'] = np.where(np.logical_and(feeder_ind.index.hour >= 6, feeder_ind.index.hour <=8),1,0)
        feeder_ind['9-11'] = np.where(np.logical_and(feeder_ind.index.hour >= 9, feeder_ind.index.hour <=11),1,0)
        feeder_ind['12-14'] = np.where(np.logical_and(feeder_ind.index.hour >= 12, feeder_ind.index.hour <=14),1,0)
        feeder_ind['15-17'] = np.where(np.logical_and(feeder_ind.index.hour >= 15, feeder_ind.index.hour <=17),1,0)
        feeder_ind['18-20'] = np.where(np.logical_and(feeder_ind.index.hour >= 18, feeder_ind.index.hour <=20),1,0)
        feeder_ind['21-23'] = np.where(np.logical_and(feeder_ind.index.hour >= 21, feeder_ind.index.hour <=23),1,0)
        
    elif hours_div == 6:
        feeder_ind['0-3'] = np.where(np.logical_and(feeder_ind.index.hour >= 0, feeder_ind.index.hour <=3),1,0)
        feeder_ind['4-7'] = np.where(np.logical_and(feeder_ind.index.hour >= 4, feeder_ind.index.hour <=7),1,0)
        feeder_ind['8-11'] = np.where(np.logical_and(feeder_ind.index.hour >= 8, feeder_ind.index.hour <=11),1,0)
        feeder_ind['12-15'] = np.where(np.logical_and(feeder_ind.index.hour >= 12, feeder_ind.index.hour <=15),1,0)
        feeder_ind['16-19'] = np.where(np.logical_and(feeder_ind.index.hour >= 16, feeder_ind.index.hour <=19),1,0)
        feeder_ind['20-23'] = np.where(np.logical_and(feeder_ind.index.hour >= 20, feeder_ind.index.hour <=23),1,0)
        
    elif hours_div == 4:
        feeder_ind['Ni'] = np.where(np.logical_and(feeder_ind.index.hour >= 0, feeder_ind.index.hour <=5),1,0)
        feeder_ind['Mo'] = np.where(np.logical_and(feeder_ind.index.hour >= 6, feeder_ind.index.hour <=11),1,0)
        feeder_ind['Af'] = np.where(np.logical_and(feeder_ind.index.hour >= 12, feeder_ind.index.hour <=17),1,0)
        feeder_ind['Ev'] = np.where(np.logical_and(feeder_ind.index.hour >= 18, feeder_ind.index.hour <=23),1,0)
        
    elif hours_div ==3:
        feeder_ind['Ni'] = np.where(np.logical_and(feeder_ind.index.hour >= 0, feeder_ind.index.hour <=7),1,0)
        feeder_ind['Mo'] = np.where(np.logical_and(feeder_ind.index.hour >= 8, feeder_ind.index.hour <=15),1,0)
        feeder_ind['Af'] = np.where(np.logical_and(feeder_ind.index.hour >= 16, feeder_ind.index.hour <=23),1,0)
        
    elif hours_div == 2:
        feeder_ind['Ni'] = np.where(np.logical_and(feeder_ind.index.hour >= 0, feeder_ind.index.hour <=11),1,0)
        feeder_ind['Da'] = np.where(np.logical_and(feeder_ind.index.hour >= 12, feeder_ind.index.hour <=23),1,0)       
    
    return  feeder_ind

def M_model(dataframe, holdout_data, mode, model_order):
    """
    Get the discrete LTI model matrix (M) for a specified model order
    
    Input parameters:
    
        dataframe (DataFrame)    data set for feeder power, temperature, and optional solar
        
        holdout_data (decimal)   fraction of data held for testing
        
        mode (string)            model mode ('hyprid' or 'dynamic')
        
        model_order (int)        model order
         
    Returns
    
        error (float)            returns mean of model error on hold out data
        
        M (matrix)               returns discrete LTI model matrix for input variables with specified model order
        
    """       
    L = model_order
    K = len(dataframe)
    N = int(len(dataframe)*holdout_data)
    T = np.matrix(dataframe.temperature).transpose()
    if mode == 'hyprid':
        P = np.matrix(dataframe.P_residual).transpose() 
    else: 
        P = np.matrix(dataframe.power).transpose()        
    if 'solar' in dataframe.columns:
        S = np.matrix(dataframe.solar).transpose()
        M = np.hstack([np.hstack([P[n:K-L+n] for n in range(L)]),
                 np.hstack([T[n:K-L+n] for n in range(L+1)]),
                 np.hstack([S[n:K-L+n] for n in range(L+1)])])
    else:
        M = np.hstack([np.hstack([P[n:K-L+n] for n in range(L)]),
                 np.hstack([T[n:K-L+n] for n in range(L+1)])])       
    Mh = M[0:-N]
    Mt = Mh.transpose()
    x = np.linalg.solve(np.matmul(Mt,Mh),np.matmul(Mt,P[L:-N]))
    Q = M[-N:]*x 
    er = (Q/P[-N:] - 1)*100
    error = er.mean()
    
    return error, M


def feature_model(dataframe, months_div, days_div, hours_div, holdout_data, mode, M, model_order):
    """
    Function to fit model with added hour/day/month features
    
    Input parameters:
    
        dataframe (DataFrame)           data set for feeder power, temperature, and optional solar with datetime index
        
        months_div (2,3,4,6,12)         months divisor in a year
        
        days_div (2,7)                  days divisor in a week
        
        hours_div (2,3,4,6,8,12,24)     hours divisor in a day
        
        holdout_data (decimal)          fraction of data held for testing
        
        mode (string)                   Model mode ('hyprid' or 'dynamic')
        
        model_order (int)               Model order
        
        M (matrix)                      discrete LTI model matrix for input variables with specified model order        
        
    Returns
    
        e_comb (matrix)                 returns model percent error for hold out data
        
        Q_comb (matrix)                 returns fitted model power data
    
        error (float)                   returns the minimized mean of model error on hold out data
        
        x_comb (matrix)                 returns fitted model coefficients 
                       
    """    
    feeder_ind = feature_selection(dataframe, months_div, days_div, hours_div)
    if mode == 'hyprid':
        P = np.matrix(dataframe.P_residual).transpose() 
    else: 
        P = np.matrix(dataframe.power).transpose()  
    P_tot = np.matrix(dataframe.power).transpose()
    P_s = np.matrix(dataframe.P_static).transpose()
    M_ind = np.matrix(feeder_ind.iloc[:,len(dataframe.columns):].values)
    N = int(len(dataframe)*holdout_data)
    M_comb = np.hstack([M,M_ind[model_order:]])
    Mh_comb = M_comb[0:-N]
    Mt_comb = Mh_comb.transpose()
    x_comb = np.linalg.solve(np.matmul(Mt_comb,Mh_comb),np.matmul(Mt_comb,P[model_order:-N]))         
    Q_comb = M_comb[-N:]*x_comb
    if mode == 'hyprid':
        e_comb = ((Q_comb + P_s[-N:])/P_tot[-N:] - 1)*100
    else: 
        e_comb = (Q_comb/P[-N:] - 1)*100       
    error = abs(e_comb).mean()
    
    return e_comb, Q_comb, error, x_comb

def best_M_order(dataframe, months_div, days_div, hours_div, holdout_data, mode, model_order):
    """
    Finds out the model order that minimizes the RMSE (error) and the corrosponding discrete LTI matrix (M)
    If model order is specified, it returns feature model function output 
    
    Input parameters:
    
        dataframe (DataFrame)           data set for feeder power, temperature, and optional solar with datetime index
        
        months_div (2,3,4,6,12)         months divisor in a year
        
        days_div (2,7)                  days divisor in a week
        
        hours_div (2,3,4,6,8,12,24)     hours divisor in a day
        
        holdout_data (decimal)          fraction of data held for testing
        
        mode (string)                   model mode ('hyprid' or 'dynamic')
        
        model_order (int or 'auto')     model order
        
    Returns
    
        e_comb (matrix)                 returns model percent error for hold out data
        
        Q_comb (matrix)                 returns fitted model power data
    
        error (float)                   returns the minimized mean of model error on hold out data
        
        x_comb (matrix)                 returns fitted model coefficients 
        
        model_order (int)               returns model order that minimizes the RMSE (error) if set to 'auto'
        
    """
    
    if model_order == 'auto':        
        Max_order = 30
        errors = []
        for i in range(1,Max_order):
            erro, M = M_model(dataframe, holdout_data, mode, model_order = i)
            e_comb, Q_comb, error, x_comb = feature_model(dataframe, months_div, days_div, hours_div, 
                                                          holdout_data, mode, M, model_order = i)

            errors.append(abs(error))            
        e = np.min(errors)
        index, = np.where(errors == e)
        m_order = index + 1
        model_order = m_order.item()
        erro, M = M_model(dataframe, holdout_data, mode, model_order) 
        e_comb, Q_comb, error, x_comb = feature_model(dataframe, months_div, days_div, hours_div, 
                                                          holdout_data, mode, M, model_order)
    else:
        erro, M = M_model(dataframe, holdout_data, mode, model_order)
        e_comb, Q_comb, error, x_comb = feature_model(dataframe, months_div, days_div, hours_div, 
                                                          holdout_data, mode, M, model_order)
    return e_comb, Q_comb, error, x_comb, model_order
        


def best_feature(dataframe, months_div, days_div, hours_div, holdout_data, mode, model_order):
    """
    Function to choose features (months_div, days_div, hours_div) that minimizes RMSE (error)
    
    Input parameters:
    
        dataframe (DataFrame)                  data set for feeder power, temperature, and optional solar with datetime index
        
        months_div (2,3,4,6,12 or 'auto')      months divisor in a year
        
        days_div (2,7 or 'auto')               days divisor in a week
        
        hours_div (2,3,4,6,8,12,24 or 'auto')  hours divisor in a day
        
        holdout_data (decimal)                 fraction of data held for testing
        
        mode (string)                          model mode ('hyprid' or 'dynamic')
        
        model_order (int)                      model order
        
        M (matrix)                             discrete LTI model matrix for input variables with specified model order        
        
    Returns
    
        e_comb (matrix)                      returns model percent error for hold out data
        
        Q_comb (matrix)                      returns fitted model power data
            
        x_comb (matrix)                      returns fitted model coefficients 
        
        months_div (int)                     months divisor in a year that minimizes RMSE (error) if set to 'auto'
        
        days_div (int)                       days divisor in a week that minimizes RMS (error) if set to 'auto'
        
        hours_div (int)                      hours divisor in a day that minimizes RMSE (error) if set to 'auto'
                        
    """
    # find the months divisor that minimizes the error
    if months_div == 'auto':
        month_divisors = [1,2,3,4,6,12]
        errors = []
        day_div = 1
        hour_div = 1
        for i in range(0,len(month_divisors)):
            months_div = month_divisors[i]
            e_comb, Q_comb, error, x_comb, model_order = best_M_order(dataframe,months_div,day_div,hour_div,
                                                                      holdout_data,mode,model_order)
            errors.append(error)
        er = np.abs(errors)
        e = np.min(er)
        index, = np.where(er == e)
        months_div = month_divisors[index.item()]        
    else:
        months_div = months_div
       
    #find the days divisor that minimizes the error
    if days_div == 'auto':
        days_divisors = [1,2,7]
        errors = []
        month_div = months_div
        hour_div = 1
        for i in range(0,len(days_divisors)):
            day_div = days_divisors[i]
            e_comb, Q_comb, error, x_comb, model_order = best_M_order(dataframe,month_div,day_div,hour_div,
                                                                      holdout_data,mode,model_order)
            errors.append(error)
        er = np.abs(errors)
        e = np.min(er)
        index, = np.where(er == e)
        days_div = days_divisors[index.item()]
    else:
        days_div = days_div
        
    # find the hours divisor that minimizes the error
    if hours_div == 'auto':
        hours_divisors = [1,2,3,4,6,8,12,24]
        errors = []
        month_div = months_div
        day_div = days_div
        for i in range(0,len(hours_divisors)):
            hour_div = hours_divisors[i]
            try:
                e_comb, Q_comb, error, x_comb, model_order = best_M_order(dataframe,month_div,day_div,hour_div,
                                                                      holdout_data,mode,model_order)
                errors.append(error)
            except:
                pass           
        er = np.abs(errors)
        e = np.min(er)
        index, = np.where(er == e)
        hours_div = hours_divisors[index.item()]
        
    else:
        hours_div = hours_div
      
    e_comb, Q_comb, error, x_comb, model_order = best_M_order(dataframe, months_div,days_div, 
                                                              hours_div, holdout_data, mode, model_order)
    
    return e_comb, Q_comb, x_comb, months_div, days_div, hours_div, model_order 


def plot_model(dataframe, e_comb, Q_comb, holdout_data, mode):
    """
    Fuction that plots model fitting on hold out data and returns
    
    Input parameters:
    
        dataframe (DataFrame)           data set for feeder power, temperature, and optional solar with datetime index
        
        e_comb (matrix)                 model percent error for hold out data
        
        Q_comb (matrix)                 fitted model power data
        
        holdout_data (decimal)          fraction of data held for testing
        
        mode (string)                   Model mode ('hyprid' or 'dynamic') 
        
    Returns:
    
        Figure 1                       error plot 
        
        Figure 2                       power for model vs. actual data
        
        Figure 3-1                     temperature vs. model and actual power data
        
        Figure 3-2                     solar vs. model and actual power data if solar was given
        
    """
    N = int(len(dataframe)*holdout_data)
    P_tot = np.matrix(dataframe.power).transpose()
    P_s = np.matrix(dataframe.P_static).transpose()
    T = np.matrix(dataframe.temperature).transpose()
    t = dataframe.index[-N:]
     
    plt.figure(1, figsize=(15,5))
    plt.plot(t,e_comb, linewidth = 0.5)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Model error %")
    plt.title(f'Error plot ({mode} model)\n Mean of error = {abs(e_comb).mean():.4f}%\nStdev of error = {abs(e_comb).std():.2f}')
    plt.show()
    
    plt.figure(2, figsize=(15,5))
    plt.plot(t,P_tot[-N:], linewidth=0.8)
    if mode == 'hyprid':
        P_model = Q_comb+P_s[-N:]
        plt.plot(t, P_model, linewidth = 0.8)
    else: 
        P_model = Q_comb
        plt.plot(t, P_model, linewidth = 0.8)     
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Power (MW)")
    plt.title("Power (Model vs Data)")
    plt.legend(["Data","Model"])
    plt.show()

    plt.figure(3,figsize=(15,5))
    plt.subplot(121)
    plt.plot(T[-N:],P_tot[-N:], linewidth=0.3)
    plt.plot(T[-N:],P_model, linewidth=0.3)
    plt.grid()
    plt.xlabel("Temperature ($^o$F)")
    plt.ylabel("Power (MW)")
    plt.title("Temperature vs Power")
    plt.legend(["Data","Model"])
    if 'solar' in dataframe.columns: 
        S = np.matrix(dataframe.solar).transpose()
        plt.subplot(122)
        plt.plot(S[-N:],P_tot[-N:], linewidth=0.3)
        plt.plot(S[-N:],P_model, linewidth=0.3)
        plt.grid()
        plt.xlabel("Solar (W/sF)")
        plt.ylabel("Power (MW)")
        plt.title("Solar gains vs Power")
        plt.legend(["Data","Model"])
    plt.show()
    
    
def dynamic_model(dataframe, months_div='auto', days_div='auto', hours_div='auto', holdout_data = 0.0275, mode = 'dynamic',
                   model_order = 'auto'):
    """
    Function to combine all functions to produce results for dynamic model fitting for load data
    
    Input parameters:
    
        dataframe (DataFrame)                  data set for feeder power, temperature, and optional solar with datetime index
        
        months_div (2,3,4,6,12 or 'auto')      months divisor in a year
        
        days_div (2,7 or 'auto')               days divisor in a week
        
        hours_div (2,3,4,6,8,12,24 or 'auto')  hours divisor in a day
        
        holdout_data (decimal)                 fraction of data held for testing (default 0.0275)
        
        mode (string)                          Model mode ('hyprid' or 'dynamic')
        
        model_order (int or 'auto')            model order, if 'auto' it finds the model order that minimizes RMSE (error)
                
    Returns
    
        Figure 1                               error plot 
        
        Figure 2                               power for model vs. actual data
        
        Figure 3-1                             temperature vs. model and actual power data
        
        Figure 3-2                             solar vs. model and actual power data if solar was given
        
        months_div                             model parameter: months divisor feature used 
        
        days_div                               model parameter: days divisor feature used 
        
        hours_div                              model parameter: hours divisor feature used 
        
        holdout_data                           model parameter: fraction of data used as hold out 
        
        tf_P_to_T                              transfer function for power to temperature
                   
    """
    e_comb, Q_comb, x_comb, months_div, days_div, hours_div, model_order = best_feature(dataframe,months_div, days_div, hours_div, 
                                                                                              holdout_data, mode,
                                                                                              model_order)
    plot_model(dataframe, e_comb, Q_comb, holdout_data, mode)
    
    print('MODEL PARAMETERS')
    print('Months divisor:', months_div)
    print('Days divisor:',days_div)
    print('Hours divisor:',hours_div)
    print('Model order:',model_order)
    print('Hold out data fraction:',holdout_data)
    
    num = np.array(x_comb[model_order:(2*model_order+1)]).flatten()
    den = np.array(x_comb[0:model_order]).flatten()
    tf_P_to_T = ct.tf(num,den,dt=1)
    
    return tf_P_to_T


# Forecaster function

def forecast(dataframe, t0, nt, mode, dt = timedelta(hours=1)):
    """
    Function to forecast power load based on historical data for power and tempeture
    
    Input parameters:
    
        dataframe (DataFrame)                  data set for feeder power and temperature with datetime index
        
        t0 (datetime(year,month,day,hour))     date and hour to start forcasting
        
        nt (int)                               number of hours to forecast ahead of t0
        
        mode ('hyprid' or 'dynamic')           get power from hyprid or dynamic model
        
    Returns:
    
        result (DataFrame)                     dataframe with forecast load for the specified time frame with datetime index   
    """
    M_order = 24       # defult model order = 24
    L = nt 
    K = len(dataframe.power)
    if mode == 'hyprid':
        P = np.matrix(dataframe.P_residual).transpose()
    elif mode == 'dynamic':
        P = np.matrix(dataframe.power).transpose()    
    T = np.matrix(dataframe.temperature).transpose()
    A = np.hstack([np.hstack([T[n:K-(M_order+L)+n] for n in range(M_order+1)]),
                   np.hstack([P[n:K-(M_order+L)+n] for n in range(M_order+L)])])
    M = A[:,0:-L]
    Mh = M
    Mt = Mh.transpose()
    Y = A[:,-L:]
    sys = np.linalg.solve(np.matmul(Mt,Mh),np.matmul(Mt,Y))
    M = M_order
    if mode == 'hyprid':
        P = dataframe.P_residual
    elif mode == 'dynamic':
        P = dataframe.power    
    T = dataframe.temperature
    i = 0    
    yy = [] 
    p = P[t0 - M*dt:t0 - dt].tolist()
    for t in pd.date_range(t0,t0+dt*(nt-1),nt):
        x = sys[:,i]
        M_forc = np.hstack([T[t0 - M*dt:t0],p])
        y = np.matmul(M_forc,x)
        yy.append(y)  
        i = i+1
    result = pd.DataFrame(np.reshape(yy,(nt,1)), pd.date_range(t0,t0+dt*(nt-1),nt), columns=['forecast'])
    result.index.name = 'datetime'
    
    return result
