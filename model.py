#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:16:12 2020

@author: ziga
"""

import numpy as np
import pandas as pd
import datetime

data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "tests.positive.todate","tests.positive","state.in_hospital",\
                                                    "state.icu","state.deceased.todate"], parse_dates=["date"])
data_stats = data_stats.fillna(0)

data_patients = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv//patients.csv",\
                         index_col="date", usecols=["date", "state.in_hospital.in","state.in_hospital.out","state.icu.in",\
                                                    "state.icu.out"], parse_dates=["date"])
data_patients = data_patients.fillna(0)


data_tests_positive_todate = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "age.female.0-4.todate",\
                                                    "age.female.5-14.todate",\
                                                    "age.female.15-24.todate",\
                                                    "age.female.25-34.todate",\
                                                    "age.female.35-44.todate",\
                                                    "age.female.45-54.todate",\
                                                    "age.female.55-64.todate",\
                                                    "age.female.65-74.todate",\
                                                    "age.female.75-84.todate",\
                                                    "age.female.85+.todate",\
                                                    "age.male.0-4.todate",\
                                                    "age.male.5-14.todate",\
                                                    "age.male.15-24.todate",\
                                                    "age.male.25-34.todate",\
                                                    "age.male.35-44.todate",\
                                                    "age.male.45-54.todate",\
                                                    "age.male.55-64.todate",\
                                                    "age.male.65-74.todate",\
                                                    "age.male.75-84.todate",\
                                                    "age.male.85+.todate"], parse_dates=["date"])   

data_tests_positive_todate = data_tests_positive_todate.fillna(0)

N = np.max(np.nonzero(data_tests_positive_todate["age.female.75-84.todate"]))
    
female_cumulative = np.zeros((9,N))
female_cumulative[0,:] = (data_tests_positive_todate["age.female.0-4.todate"] + data_tests_positive_todate["age.female.5-14.todate"])[:N]/2.
female_cumulative[1,:] = (data_tests_positive_todate["age.female.5-14.todate"] + data_tests_positive_todate["age.female.15-24.todate"])[:N]/2.
female_cumulative[2,:] = (data_tests_positive_todate["age.female.15-24.todate"] + data_tests_positive_todate["age.female.25-34.todate"])[:N]/2.
female_cumulative[3,:] = (data_tests_positive_todate["age.female.25-34.todate"] + data_tests_positive_todate["age.female.35-44.todate"])[:N]/2.
female_cumulative[4,:] = (data_tests_positive_todate["age.female.35-44.todate"] + data_tests_positive_todate["age.female.45-54.todate"])[:N]/2.
female_cumulative[5,:] = (data_tests_positive_todate["age.female.45-54.todate"] + data_tests_positive_todate["age.female.55-64.todate"])[:N]/2.
female_cumulative[6,:] = (data_tests_positive_todate["age.female.55-64.todate"] + data_tests_positive_todate["age.female.65-74.todate"])[:N]/2.
female_cumulative[7,:] = (data_tests_positive_todate["age.female.65-74.todate"] + data_tests_positive_todate["age.female.75-84.todate"])[:N]/2.
female_cumulative[8,:] = (data_tests_positive_todate["age.female.75-84.todate"] + data_tests_positive_todate["age.female.85+.todate"])[:N]/2.

female_daily = np.zeros((9,N+21))
female_daily[:,1:N] = female_cumulative[:,1:] - female_cumulative[:,:-1]


male_cumulative = np.zeros((9,N))
male_cumulative[0,:] = (data_tests_positive_todate["age.male.0-4.todate"] + data_tests_positive_todate["age.male.5-14.todate"])[:N]/2.
male_cumulative[1,:] = (data_tests_positive_todate["age.male.5-14.todate"] + data_tests_positive_todate["age.male.15-24.todate"])[:N]/2.
male_cumulative[2,:] = (data_tests_positive_todate["age.male.15-24.todate"] + data_tests_positive_todate["age.male.25-34.todate"])[:N]/2.
male_cumulative[3,:] = (data_tests_positive_todate["age.male.25-34.todate"] + data_tests_positive_todate["age.male.35-44.todate"])[:N]/2.
male_cumulative[4,:] = (data_tests_positive_todate["age.male.35-44.todate"] + data_tests_positive_todate["age.male.45-54.todate"])[:N]/2.
male_cumulative[5,:] = (data_tests_positive_todate["age.male.45-54.todate"] + data_tests_positive_todate["age.male.55-64.todate"])[:N]/2.
male_cumulative[6,:] = (data_tests_positive_todate["age.male.55-64.todate"] + data_tests_positive_todate["age.male.65-74.todate"])[:N]/2.
male_cumulative[7,:] = (data_tests_positive_todate["age.male.65-74.todate"] + data_tests_positive_todate["age.male.75-84.todate"])[:N]/2.
male_cumulative[8,:] = (data_tests_positive_todate["age.male.75-84.todate"] + data_tests_positive_todate["age.male.85+.todate"])[:N]/2.

male_daily = np.zeros((9,N+21))
male_daily[:,1:N] = male_cumulative[:,1:] - male_cumulative[:,:-1]    

#%% extrapolate numbers for next 21 days from last 14 days
# compute linear trend for each category
from scipy.optimize import curve_fit

L_decay = True
decay_slope = 0.1 # weekly decay
L_linear_trend = False
L_average = False

if L_decay:
    lambd = - np.log(1.-decay_slope)/7
    female_mean = female_daily[:,N-7:N].mean(axis=1) 
    male_mean = male_daily[:,N-7:N].mean(axis=1)
    
    d = 0
    for i in range(N,N+21):
        d += 1
        female_daily[:,i] = female_mean*np.exp(-lambd*d)
        male_daily[:,i] = male_mean*np.exp(-lambd*d)
    

elif L_average:
    female_mean = female_daily[:,N-14:N].mean(axis=1) 
    male_mean = male_daily[:,N-14:N].mean(axis=1) 
    for i in range(N,N+21):
        female_daily[:,i] = female_mean
        male_daily[:,i] = male_mean
        
elif L_linear_trend:
    
    

#%% risks

ages = np.array([5,15,25,35,45,55,65,75,85])

risk_hosp = np.array([3.2, 1.1, 2.2, 3.8, 5.8, 9.2, 18.5, 33.3, 34.2])/100.
risk_hosp_F = np.array([2.8, 1.1, 2.3, 3.7, 4.4, 6.6, 13.8, 27.5, 28.5]) /100.
risk_hosp_M = np.array([3.5, 1.2, 2.1, 4.0, 7.3, 12.1, 22.6, 38.4, 44.3])/100.

risk_hosp_severe = np.array([0.1, 0.1, 0.1, 0.2, 0.6, 1.3, 3.5, 5.2, 2.9])/100.
risk_hosp_severe_F = np.array([0.1, 0.04, 0.1, 0.2, 0.3, 0.6, 1.9, 3.4, 2.1])/100.
risk_hosp_severe_M = np.array([0.1, 0.1, 0.1, 0.3, 0.9, 2.1, 4.9, 6.9, 4.3])/100.

risk_hosp_normal_F = risk_hosp_F - risk_hosp_severe_F
risk_hosp_normal_M = risk_hosp_M - risk_hosp_severe_M

risk_cfr = np.array([0,0,0,0.1,0.2,0.7,3.3,12.,23.3])/100.
risk_cfr_F = np.array([0,0,0,0.1,0.1,0.3,1.8,8.1,19])/100.
risk_cfr_M = np.array([0,0,0,0.1,0.3,1.1,4.6,15.3,30.7])/100.

#%% delays
from scipy import stats

ihd1,ihd2,ihd3 = 0.9405341363035055, 0, 8.996686922156094
ihls1, ihls2, ihls3 = 0.859025, 0, 9. + 2.
ihln1, ihln2, ihln3 = 0.859025, 0, 8.991630
ihicu1, ihicu2, ihicu3 = 1.6628700590950236, 0, 1.0039836228453107
th1,th2,th3 = 0.859025, 0, 1.


#stats.lognorm.rvs()
#%%

hospitalised_in = np.zeros(N+1000)
hospitalised_out = np.zeros(N+1000)
icu_in = np.zeros(N+1000)
icu_out = np.zeros(N+1000)
deaths = np.zeros(N+1000)


day0 = datetime.datetime(2020,2,24)
dayend = day0 + datetime.timedelta(days=int(N+21))
day = day0
t= 0
while day < dayend:
    print(t,day.strftime("%Y%m%d"))
    # female
    for i in  range(9):
        for k in range(int(np.round(female_daily[i,t]+np.random.uniform(-0.05,0.05),0))):
            
            test2hosp = stats.lognorm.rvs(th1,th2,th3)
            test2icu = test2hosp + stats.lognorm.rvs(ihicu1, ihicu2, ihicu3)
            
            
            fin_test2hospin = np.min((int(np.round(test2hosp,0)),21))
            fin_test2hospout = np.min((int(np.round(test2hosp + stats.lognorm.rvs(ihln1, ihln2, ihln3),0)),80))
            fin_test2icuin = np.min((int(np.round(test2icu,0)),30))
            fin_test2icuout = np.min((int(np.round(test2icu+ stats.lognorm.rvs(ihls1, ihls2, ihls3),0)),80))
            fin_test2death = np.min((int(np.round(stats.lognorm.rvs(ihd1,ihd2,ihd3),0)),100))
            
            outcome_hosp = np.random.choice([2,1,0],\
                                p=[risk_hosp_severe_F[i],risk_hosp_normal_F[i],1-risk_hosp_severe_F[i]-risk_hosp_normal_F[i]])
            
            hospitalised_in[t+fin_test2hospin] += (outcome_hosp==1)
            hospitalised_out[t+fin_test2hospout] += (outcome_hosp==1)
            icu_in[t+fin_test2icuin] += (outcome_hosp==2)
            icu_out[t+fin_test2icuout] += (outcome_hosp==2)
            deaths[t+fin_test2death] += np.random.choice([1,0],p=[risk_cfr_F[i],1-risk_cfr_F[i]])
        
    # male
    for i in range(9):
        for k in range(int(np.round(male_daily[i,t]+np.random.uniform(-0.05,0.05),0))):
            test2hosp = stats.lognorm.rvs(th1,th2,th3)
            test2icu = test2hosp + stats.lognorm.rvs(ihicu1, ihicu2, ihicu3)
            
            
            fin_test2hospin = np.min((int(np.round(test2hosp,0)),21))
            fin_test2hospout = np.min((int(np.round(test2hosp + stats.lognorm.rvs(ihln1, ihln2, ihln3),0)),60))
            fin_test2icuin = np.min((int(np.round(test2icu,0)),30))
            fin_test2icuout = np.min((int(np.round(test2icu+ stats.lognorm.rvs(ihls1, ihls2, ihls3),0)),60))
            fin_test2death = np.min((int(np.round(stats.lognorm.rvs(ihd1,ihd2,ihd3),0)),100))
            
            outcome_hosp = np.random.choice([2,1,0],\
                                p=[risk_hosp_severe_M[i],risk_hosp_normal_M[i],1-risk_hosp_severe_M[i]-risk_hosp_normal_M[i]])
            
            hospitalised_in[t+fin_test2hospin] += (outcome_hosp==1)
            hospitalised_out[t+fin_test2hospout] += (outcome_hosp==1)
            icu_in[t+fin_test2icuin] += (outcome_hosp==2)
            icu_out[t+fin_test2icuout] += (outcome_hosp==2)
            deaths[t+fin_test2death] += np.random.choice([1,0],p=[risk_cfr_M[i],1-risk_cfr_M[i]])
        
        
    day = day + datetime.timedelta(days=1)
    t += 1


#%% compare to data
import matplotlib.pyplot as plt

data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "tests.positive.todate","tests.positive","state.in_hospital",\
                                                    "state.icu","state.deceased.todate"], parse_dates=["date"])

dates = [day0 + datetime.timedelta(days=i) for i in range(400)] 

    
icu_data = data_stats["state.icu"].values
hos_data = data_stats["state.in_hospital"].values
cum_deaths_data = data_stats["state.deceased.todate"].values


nn= icu_data.shape[0]
deaths_data = np.zeros(nn)
deaths_data[1:] = cum_deaths_data[1:] - cum_deaths_data[:-1]


plt.figure(1,figsize=(12,6))
plt.plot_date(dates[:nn],icu_data,"ro",markersize=2,label="ICU data",xdate=True)
plt.plot_date(dates[:nn],hos_data,"bo",markersize=2,label="HOSP data",xdate=True)
plt.plot_date(dates[:nn],cum_deaths_data,"ko",markersize=2,label="deaths data",xdate=True)

plt.plot_date(dates[:nn+20],np.cumsum(icu_in-icu_out)[:nn+20],"r-",xdate=True,label="ICU model")
plt.plot_date(dates[:nn+20],np.cumsum(hospitalised_in-hospitalised_out)[:nn+20],"b-",xdate=True,label="HOSP model")
plt.plot_date(dates[:nn+20],np.cumsum(deaths)[:nn+20],"k-",xdate=True,label="deaths model")

plt.plot_date(dates[:nn],deaths_data,"go",markersize=2,label="daily deaths data",xdate=True)
plt.plot_date(dates[:nn+20],(deaths)[:nn+20],"g-",xdate=True,label="daily deaths model")

plt.yscale("log")
plt.xticks(rotation=45)
plt.grid()
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
plt.legend()

#%%

hosp_in_data = data_patients["state.in_hospital.in"].values
hosp_out_data = data_patients["state.in_hospital.out"].values
icu_in_data = data_patients["state.icu.in"].values
icu_out_data = data_patients["state.icu.out"].values


plt.figure(2,figsize=(12,6))


plt.plot_date(dates[9:nn],hosp_in_data,"ro",markersize=2,label="HOSP-IN data",xdate=True)
plt.plot_date(dates[9:nn],hosp_out_data,"go",markersize=2,label="HOSP-OUT data",xdate=True)
plt.plot_date(dates[9:nn],icu_in_data,"ko",markersize=2,label="ICU-IN data",xdate=True)
plt.plot_date(dates[9:nn],icu_out_data,"bo",markersize=2,label="ICU-OUT data",xdate=True)

plt.plot_date(dates[:nn+20],hospitalised_in[:nn+20],"r-",xdate=True,label="hosp-in model")
plt.plot_date(dates[:nn+20],hospitalised_out[:nn+20],"g-",xdate=True,label="hosp-out model")

plt.plot_date(dates[:nn+20],icu_in[:nn+20],"k-",xdate=True,label="icu-in model")
plt.plot_date(dates[:nn+20],icu_out[:nn+20],"b-",xdate=True,label="icu-out model")

plt.yscale("log")
plt.xticks(rotation=45)
plt.grid()
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
plt.legend()


#%% prediction

icu_model = np.cumsum(icu_in-icu_out)[:nn+20]
hosp_model = np.cumsum(hospitalised_in-hospitalised_out)[:nn+20]
deaths_model = np.cumsum(deaths)[:nn+20]

plt.figure(3,figsize=(12,6))
plt.plot_date(dates[:nn],icu_data,"ro",markersize=2,label="ICU data",xdate=True)
plt.plot_date(dates[:nn],hos_data,"bo",markersize=2,label="HOSP data",xdate=True)
plt.plot_date(dates[:nn],cum_deaths_data,"ko",markersize=2,label="deaths data",xdate=True)

plt.plot_date(dates[:nn+20],icu_model*np.mean(icu_data[nn-5:nn])/np.mean(icu_model[nn-5:nn]),"r-",xdate=True,label="ICU model")
plt.plot_date(dates[:nn+20],hosp_model*np.mean(hos_data[nn-5:nn])/np.mean(hosp_model[nn-5:nn]),"b-",xdate=True,label="HOSP model")
plt.plot_date(dates[:nn+20],deaths_model*np.mean(cum_deaths_data[nn-5:nn])/np.mean(deaths_model[nn-5:nn]),"k-",xdate=True,label="deaths model")


plt.plot_date(dates[:nn],deaths_data,"go",markersize=2,label="daily deaths data",xdate=True)
plt.plot_date(dates[:nn+20],(deaths)[:nn+20],"g-",xdate=True,label="daily deaths model")

plt.yscale("log")
plt.xticks(rotation=45)
plt.grid()
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
plt.xlim([datetime.date(2020, 11, 10),datetime.date(2020, 12, 20)])
plt.legend()
plt.ylim([10,3000])


