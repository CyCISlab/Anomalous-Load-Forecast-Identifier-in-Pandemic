import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import math


#Preparing the Data
def prepare_data(path):
    os.chdir(path)
    extension = 'xlsx'
    files = [i for i in glob.glob('*.{}'.format(extension))] 
    ''' Getting the data from the each file in the Load Folder '''
    load_df = pd.DataFrame()
    for file in files:
        file_df = pd.read_excel((path + file), header=None,index_col=0, dtype={'Name': str, 'Value': float})
        load_df=pd.concat([load_df, file_df[1]], axis=1)
    load_df.columns = [file[:-5] for file in files]    
    return load_df

#Optimal Value Calculation of eta (no of days of historical data)
def opt_eta_deter(data_df, eta_start, eta_end,learn_start_day,learn_end_day):
    avg_MAPE=[]
    for eta in range(eta_start, eta_end):
        eta_factor_sum = 0 # eta Factor Sum
    
        #eta Factor Sum
        for j in range(1, eta+1):
            eta_factor_sum+=pow(eta, eta-j)
        
        #Weights Matrix
        wt=[]
        for i in range(1, eta+1):
            wt.append((pow(eta,(eta-i))/eta_factor_sum))

        #print(wt)
        MAPE = []

        #print(wt)
        #observing optimal eta value for N observation days
        for D in range(learn_start_day, learn_end_day+1):
            mape_inter_sum = 0     
            num_intervals = data_df[:,(D-1)].size
            Y = [] 
            for t in range(0, num_intervals):
                #Forecasting Formula
                Ytd=0
                for i in range(1, eta+1):
                    Ytd+=wt[i-1] * data_df[t][D-i-1]   
                Y.append((Ytd))
                mape_inter_sum+= abs((Ytd - data_df[t][D-1])/data_df[t][D-1])
            #calculating MAPE
            MAPE.append((100*mape_inter_sum/num_intervals))
            #print(Y)
            
        #Average MAPE
        avg_MAPE.append(np.sum(MAPE)/(learn_end_day-learn_start_day))
    #Optimum eta Value
    opt_eta_val = np.where(avg_MAPE == np.amin(avg_MAPE))
    return avg_MAPE, opt_eta_val[0]+eta_start

  
#Forecasting Day Ahead Load    
def forecast_load(data_df, order_df, opt_eta,intervals,forecast_day,max_load_dev):
    day_ahead_load=[]
    opt_eta_sum = 0 # eta Factor Sum
    for j in range(1, opt_eta+1):
        opt_eta_sum+=pow(opt_eta, opt_eta-j)
        
    #Weights Matrix
    wt=[]
    for i in range(1, opt_eta+1):
        wt.append((pow(opt_eta,(opt_eta-i))/opt_eta_sum))
    for t in range(intervals):
        Ytd=0
        for i in range(opt_eta):
            Ytd+=wt[i]*data_df[t][forecast_day-i-2]
        if (order_df[t][forecast_day-1] != 0):
            b=0
            if order_df[t][forecast_day-2] == 0:
                b = -0.2/(math.pow(2, (order_df[t][forecast_day-1]-1)))
            elif (order_df[t][forecast_day-2] > (order_df[t][forecast_day-1])):
                b = -0.2/(math.pow(2, (order_df[t][forecast_day-2] - order_df[t][forecast_day-1])))
            elif (order_df[t][forecast_day-2] < (order_df[t][forecast_day-1])):
                b = 0.2/(math.pow(2, (order_df[t][forecast_day-1] - order_df[t][forecast_day-2])))
            Ytd = (1+b)*Ytd #Incorporation of the covid effect
        day_ahead_load.append((Ytd))
    return day_ahead_load


#MultiDay Load forecasting
def multi_forecast(data_df,order_df, opt_eta, interval, forecast_start, forecast_days,max_load_dev):
    multi_forecast_mat = np.empty((0,interval), float)
    for day in range(forecast_days):
        day_forecast = forecast_load(data_df,order_df, opt_eta,interval,forecast_start+day,max_load_dev)
        multi_forecast_mat = np.vstack((multi_forecast_mat, day_forecast))
        if(np.size(data_df, 1) <= day+forecast_start):
            data_df = np.column_stack((data_df,day_forecast))
    pd.DataFrame(data_df).to_excel('Forcasting/load_matrix.xlsx',index=False)
    return  multi_forecast_mat


def daterange(start_date, num):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    daterange = [(start + timedelta(days=x)).date().strftime("%Y-%m-%d") for x in range(0, num)]
    return daterange



def Perform_forcast(load_data_folder_path, order_related_severity_matrix_path, forecast_start_date, forecast_days, start_date, end_date, eta_start, eta_end, interval):
    
    #Constants - 
    max_load_dev = 0.2
    #Data Folder Path 
    path = load_data_folder_path

    #Prepare data and Store data to excel
    load_df = prepare_data(path)
    load_df.to_excel('Forcasting/final_data.xlsx')
    load_np_arr = load_df.to_numpy()
    
    #Order Related Severity matrix
    order_path = order_related_severity_matrix_path
    order_df = pd.read_excel(order_path, index_col=None, dtype={'Name': str, 'Value': float})
    order_np_arr = order_df.to_numpy()
    

    #Forecast Interval -
    forecast_start_date = forecast_start_date
    forecast_days = int(forecast_days)
    
    #Optimal eta Value Determination
    start_date = start_date  #='2021-05-09'  #'YY-MM-DD' 
    end_date =  end_date #='2021-05-14'

    eta_start = int(eta_start)
    eta_end = int(eta_end)
    interval = int(interval)

    cols = load_df.columns.tolist()

    if (start_date == "" or end_date == ""):
        if forecast_start_date in cols:
            forecast_date_index = cols.index(forecast_start_date)+1
            learn_start_day = forecast_date_index-6
            learn_end_day = forecast_date_index-1
            if forecast_date_index <= learn_end_day:
                print('Forecast Date is below or equal to Learn End Date.')
        else:
                forecast_date_index = len(cols)+1
                learn_start_day = forecast_date_index-6
                learn_end_day = forecast_date_index-1

    else: 
        learn_start_day = cols.index(start_date) + 1
        learn_end_day = cols.index(end_date) + 1  
        if forecast_start_date in cols:
            forecast_date_index = cols.index(forecast_start_date)+1
            if forecast_date_index <= learn_end_day:
                print('Forecast Date is below or equal to Learn End Date.')
        else:
            forecast_date_index = len(cols)+1 

    #determination of optimal eta
    avg_MAPE,opt_eta = opt_eta_deter(load_np_arr, eta_start, eta_end,learn_start_day,learn_end_day)


    #Forecast -- 
    
    multi_forecast_mat = multi_forecast(load_np_arr,order_np_arr, int(opt_eta),interval, forecast_date_index, forecast_days,max_load_dev)
    forecast_df = pd.DataFrame(multi_forecast_mat)
    forecast_df.index = daterange(forecast_start_date,forecast_days)
    forecast_df.to_excel('Forcasting/forecast_data.xlsx')
    print(multi_forecast_mat.size,"END\n")
    

if __name__=="__main__":
    main()