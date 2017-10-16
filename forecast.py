import pandas as pd
import numpy as np
import sys
from pandas.tseries.holiday import USFederalHolidayCalendar
import plac
import random
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib
# for interactive charts in ipython...(we can use chartly as well..
#import plotly.plotly as py
import json
import time
from time import gmtime, strftime
import os


# for ipython; set some defaults for the charts
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('ggplot')


input_dir = "data"
output_dir = "outputs"

mdf = pd.read_csv(os.path.join(input_dir,"agg_date_zip.csv"))
del mdf["Unnamed: 0"]


def predictZip(zipcode, 
               main_df=None,
               duration     = 182,
               start        = '2014-01-01',
               end          = '2015-12-31',
               plot         = False,
               to_log       = False,
               growth_model = "logistic",
               cps          = 0.005,
               train        = True):
    """
     zip code is a string representation; number might not work..
     main_df contains the data for all zip codes and by date..
     start, end: format is "2017-12-31" duration: in days
     plot: boolean for showing pretty charts; this works on ipython 
     and notebooks only,  else might throw and exception
     to_log: boolean indicating to take log values of the y or not; 
     this can also be determined by looking at variance
     returns a type of prophet model instance and forecasted df and 
     accuracy df with orig and forecasted values

    """

    if not main_df:
        main_df = mdf

    #growth_model = "logistic"
    #Get the aggregated cobroke_total for each zip code and date    
    start_time = time.time()
    df         = main_df[(main_df["zip"] == zipcode)].groupby(["date", "zip"]).sum()
    df         = df.reset_index(level=[0,1])[["date", "cobroke_total"]]
    #plot the aggregated leads per day for each zip code   
    if plot:
        df.plot(title="Leads per day for zipcode: {}".format(zipcode), figsize=(18,8))


    df["date"]    = pd.to_datetime(df["date"],infer_datetime_format=True)

    # prepare the df for prediction and for plotting
    pdf           = df[(df['date'] >= start) & (df['date'] <= end)]
    pdf           = pdf.rename (columns={"date":"ds", "cobroke_total":"y"})
    pdf["y_orig"] = pdf["y"] # storing orig values for charting later..
    # Take log on the data to smoothen the curve
    if to_log:
        pdf["y"]  = np.log (pdf["y"])

    pdf["y_log"]  = pdf ["y"]
    # Determine the carrying capacity for this zip code, based on census, housing stats  and move's historic performance at this zip code
    cap           = getCap(zipcode, pdf)
    if not (growth_model == "linear"):
        pdf["cap"]    = cap
    # Model the holidays
    cal = USFederalHolidayCalendar () 
    us_holidays = pd.DataFrame ({
        'holiday'     : 'federal_holidays',
        'ds'          : cal.holidays (start="2014-01-01", end="2017-12-31"),
        'lower_window': 0,
        'upper_window': 1,
    })

    
    # We are using the logistic growth model; with a lot more constrained change points
    m = Prophet(growth                  = growth_model, 
                changepoint_prior_scale = cps, 
                yearly_seasonality      = True, 
                weekly_seasonality      = True,
                holidays                = us_holidays )
    m.fit(pdf)
    future                              = m.make_future_dataframe (periods=duration)
    if not (growth_model == "linear"):
        future["cap"]                   = cap
    forecast                            = m.predict(future)
    # Plot the forecated co-broke leads 
    if plot:
        m.plot(forecast, uncertainty = True,  xlabel="Time", ylabel="Total Co-Broke Leads for zip: {}".format(zipcode))
        plot_df  = pd.concat ([df["date"], df['cobroke_total'], forecast['yhat']], axis=1, keys=['date','orig', 'forecast'])
        plot_df1 = plot_df[(plot_df['date'] >= start) & (plot_df['date'] <= end)]
        
        print ("YHAT total: {} orig total: {}".format(plot_df1["forecast"].sum(), plot_df1["orig"].sum()))
        plot_df.set_index("date", inplace=True)
        plot_df.plot(figsize=(18,8))
        if not train:
            m.plot_components(forecast)

    print("Completed forecasting for zip {}, duration: {}, in {}secs with {} growth model".format(zipcode, duration, time.time()-start_time, growth_model))

    adf = None

    # If the prediction is called just for training...
    if (train): 
        adf, a = computeAccuracy (df[(df['date'] > end)], forecast[forecast["ds"] > end], to_log)
        if np.abs(a) <= 15:
            with open(os.path.join(input_dir,"zips_acc.txt"), "a") as f:
                f.write("{},{},{}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), zipcode, a))
        if plot:
            adf.set_index("date", inplace=True)
            adf[["orig", "forecast"]].plot(figsize=(18,8))

        return (m, forecast, adf, a)
    else:
        f = forecast[forecast["ds"] > end]
        plot_df = pd.concat([f["ds"],f["yhat"]], axis = 1, keys=['date','yhat'])
        if (to_log):
            plot_df["yhat_e"] = np.exp(plot_df["yhat"])
        xx = plot_df["yhat_e"].sum()
        print ("Forecast for {} days: Total Co-broke Leads : {}".format(len(plot_df), xx))

        return (m, forecast, None, None)


def computeAccuracy (orig, forecast, to_log):
    # Function to compute accuracy and calculate error % using original leads and forecasted leads.
    df             = pd.concat ([forecast["ds"], orig['cobroke_total'], forecast['yhat']], axis=1, keys=['date','orig', 'forecast'])
    if (to_log):
        df["forecast"] = np.exp(df["forecast"])
    df["accuracy"] = (df["forecast"] - df["orig"]) / df["orig"] * 100
    o_sum          = df["orig"].sum()
    f_sum          = df["forecast"].sum()
    a              = np.round(((f_sum - o_sum) / o_sum) * 100, 2)
    print ("Accuracy: {}%, Error Rate {}% orig {}, forecast {}".format(100-np.abs(a),a, o_sum, f_sum))
        
    return df, a


def getCap(zipcode, df):
    # This needs to model with the census, housing stats, sold and zip code level information and historic performance of move
    # at this zipcode to determined the carrying capacity
    # for now..will ceil it to 20% growth on the max they have seen in the past 2 years

    xx = np.round(df["y"].max() * 1.8)
    print ("cap {}, mean {}, max {}".format(xx, df["y"].mean(), df["y"].max()))
    return xx


def trainTop (num, gm ="logistic"):
    """
    Forecast for the top 'n', zipcodes. If n == 0 then forecast for all zipcodes

    """
    import csv
    
    if type(num) == str:
        num = int(num)
    #top_zips.csv has aggregated leads data based on zips     
    zdf                = pd.read_csv(os.path.join(input_dir,"top_zips.csv"))
    done_df            = pd.read_csv(os.path.join(input_dir,"results.csv"))
    done_df["zipcode"] = done_df["zipcode"].astype(str)
    xdf                = zdf[~zdf["zip"].isin(done_df["zipcode"].tolist())]
    topzips            = xdf["zip"].head(num).tolist()
    
    # write the output for the top n zipcodes
    with open('data/results.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for zip in topzips:
            try:
                retList = trainZip(zip)
                print ('{}'.format(retList))
                csvwriter.writerow(retList)
                csvfile.flush()
            except:
                csvwriter.writerow([zip,'FAIL','ERROR',0,0,0])

    
def predictTop (num, gm ="logistic"):
    """
    Forecast for the top 'n', zipcodes. If n == 0 then forecast for all zipcodes

    """
    if type(num) == str:
        num = int(num)
        
    zdf = pd.read_csv("data/top_zips.csv")
    topzips = zdf["zip"].head(num).tolist()
    
    for zips in topzips:
        predictZip(zips)

        

def predictAgg ( growth_model = "logistic",
                 duration     = 182,
                 start        = '2014-01-01',
                 end          = '2015-12-31',
                 plot         = False,
                 to_log       = True,
                 cps          = 0.002,
                 cap_per      = 1.2

):
    """
    Predict aggregated leads from 

    """
    start_time = time.time()
    
    #Get the zip codes having leads greater than 14 for 6 months
    df    = pd.read_csv(os.path.join(input_dir,"agg_date_zip.csv"))   
    tz_df = df[(df['date'] >= "2016-01-01") & (df['date'] <= "2016-06-30")].groupby("zip").sum()
    tz_df = tz_df[tz_df["cobroke_total"] >= 14]


    df        = df[df.zip.isin(list(tz_df.index.values))]
    ds        = df.groupby("date")["cobroke_total"].sum()

    pdf       = pd.DataFrame({"ds":ds.index, "y":ds.values})
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf       = pdf[(pdf['ds'] >= start) & (pdf['ds'] <= end)]
    if to_log:
        pdf["y_orig"] = pdf["y"] # storing orig values for charting later..
        pdf["y"]      = np.log (pdf["y"])
        pdf["y_log"]  = pdf["y"]
        
    if not (growth_model == "linear"):
        cap = np.round(pdf["y"].max() * cap_per)
        print("Cap: {}".format(cap))
        pdf["cap"]    = cap

    # Model the holidays
    pdf["cap"]    = cap
    cal           = USFederalHolidayCalendar() 
    us_holidays   = pd.DataFrame ({
        'holiday'     : 'federal_holidays',
        'ds'          : cal.holidays(start="2014-01-01", end="2017-12-31"),
        'lower_window': 0,
        'upper_window': 1,
    })

    # We are using the logistic growth model; with a lot more constrained change points 
    m = Prophet(growth                  = growth_model, 
                changepoint_prior_scale = cps, 
                yearly_seasonality      = True, 
                weekly_seasonality      = True, 
                holidays                = us_holidays )
    m.fit(pdf)

    #Forecast leads for the given duration
    future                              = m.make_future_dataframe (periods=duration)
    if not (growth_model == "linear"):
        future["cap"]                   = cap
    forecast                            = m.predict(future)
    #Plot the forecast if the function called with plot=True
    if plot:
        m.plot(forecast, uncertainty = True,  xlabel="Time", ylabel="Total Co-Broke Leads for across Zipcodes")
        plot_df  = pd.concat ([df["date"], df['cobroke_total'], forecast['yhat']], axis=1, keys=['date','orig', 'forecast'])
        plot_df1 = plot_df[(plot_df['date'] >= start) & (plot_df['date'] <= end)]

        plot_df1 = plot_df[(plot_df['date'] > end)]


        if to_log:
            o_sum = np.log(plot_df1["orig"].sum())
        else:
            o_sum = plot_df1["orig"].sum()
        print ("YHAT total: {} orig total: {}".format(plot_df1["forecast"].sum(), o_sum))
        plot_df.set_index("date")
        plot_df.plot(figsize=(18,8))
        m.plot_components(forecast)

    print("Completed forecasting for agg, duration: {}, in {}secs with {} growth model".format(duration, time.time()-start_time, growth_model))

    adf = None
    if (duration < 185): # this based on start and end dates
        # handle this when pdf is given as a param; by default its none..then read from that than df
        adf, a = computeAccuracy (df[(df['date'] > end)], forecast[forecast["ds"] > end], to_log)
        if plot:
            adf[["orig", "forecast"]].plot(figsize=(18,8))
    else:
        f = forecast[forecast["ds"] > end]
        plot_df = pd.concat([f["ds"],f["yhat"]], axis = 1, keys=['date','yhat'])
        if (to_log):
            plot_df["yhat_e"] = np.exp(plot_df["yhat"])
        xx = plot_df["yhat_e"].sum()
        print ("Forecast for {} days: Total Cobroke Leads : {}".format(len(plot_df), xx))
        adf = plot_df
    return (m, forecast, adf)
    


def trainZip (zipcode):
    
    #Train the model for the given zipcode using the initial parameters below 
    tolerance_limit = 10
    max_runs = 20
    cps_step = 0.1
    runs = []
    max_accuracy = 0

    for growth in ['linear', 'logistic']:
        broken = False
        cps = 0.05
        for i in range(max_runs):
            m,f,ad,error_rate = predictZip(zipcode, growth_model=growth, cps=cps)
            accuracy = 100 - abs(error_rate)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_run   = len(runs)

            runs.append({
                "zipcode" : zipcode,
                "error_rate" : error_rate,
                "model" : m,
                "forecast" : f,
                "accuracy": accuracy,
                "accuracyd_df":ad,
                "run" : i,
                "growth" : growth,
                "cps" : cps
            })

            # if its the accuracy is < target and we are generating a higher number
            # reduce the cps param by 10 % and try

            if (error_rate) > 0:
                cps = cps * (1 + cps_step)
            else:
                cps = cps * (1 - cps_step)

            if abs(error_rate) <= tolerance_limit:
                broken = True
                break

            if abs(error_rate) >= 20:
                break

        if broken:
            break
            
    print("Max accuracy for zip {} is {}, growth model: {}".format(zipcode, runs[max_accuracy_run]['accuracy'], runs[max_accuracy_run]['growth']))

    max_accuracy_run_data = runs[max_accuracy_run]

    result = 'FAIL'
    if (100 - max_accuracy_run_data['accuracy']) <= tolerance_limit:
        result = 'PASS'

    return [zipcode, result, max_accuracy_run_data['error_rate'], max_accuracy_run_data['accuracy'], max_accuracy_run_data['growth'],max_accuracy_run_data['cps']]

df = pd.read_csv(os.path.join(input_dir,"agg_date_zip.csv"))
del df["Unnamed: 0"]

ds    = df.groupby("date")["cobroke_total"].sum()
tz    = pd.read_csv(os.path.join(input_dir,"top_zips.csv"))
df_2016_zips = df[(df['date'] >= "2016-01-01") & (df['date'] <= "2016-06-30")].groupby("zip").sum()


def predictByGroup (start_index   = 0,
                    end_index     = 1000,
                    filter_list   = None,
                    to_remove_lst = None,
                    cps           = 0.05,
                    cap_per       = 1.8,
                    growth_model  = "logistic",
                    test          = True,
                    yhat_lower    = False,
                    fcast_total   = None ):

    start_time   = time.time()
    to_log       = False

    
    if test:
        # one observation higher than the last date for training set
        end_ts       = "2016-01-01"
        forecast_end = "2016-06-30"
        duration     = 182
    else:
        end_ts       = "2016-07-01"
        forecast_end = "2017-07-31"
        duration     = 365
        
        
    if filter_list:
        tz_df = df[df.zip.isin(filter_list)]    
    else:
        xx = tz[start_index:end_index]["zip"].tolist()
        if to_remove_lst:
            xx = list(set(xx) - set(to_remove_lst))
        # slice the zips to the required segment 
        tz_df = df[df.zip.isin(xx)]    

    
    # get the complete training data for the requested segment
    tz_ds = tz_df[(tz_df['date'] >= "2015-01-01") & (tz_df['date'] <= "2015-06-30")].groupby("zip").sum()
    
    # get the sum of the co-broke for each zip...this is used to determine the distribution of the
    # zipcodes w.r.t. the total sum.
    total            = tz_ds["cobroke_total"].sum()
    tz_ds["percent"] = tz_ds["cobroke_total"] / total
    
    if fcast_total is None:
        agg_ds    = tz_df[tz_df['date'] < end_ts].groupby("date")["cobroke_total"].sum()

        pdf       = pd.DataFrame({"ds":agg_ds.index, "y":agg_ds.values})
        pdf['ds'] = pd.to_datetime(pdf['ds'])


        if to_log:
            pdf["y"]  = np.log (pdf["y"])

        if not (growth_model == "linear"):
            cap = np.round(pdf["y"].max() * cap_per)
            pdf["cap"]    = cap

        cal           = USFederalHolidayCalendar() # Model the holidays
        us_holidays   = pd.DataFrame ({
            'holiday'     : 'federal_holidays',
            'ds'          : cal.holidays(start="2014-01-01", end="2017-12-31"),
            'lower_window': 0,
            'upper_window': 1,
        })

        # We are using the logistic growth model; with a lot more constrained change points 
        m = Prophet(growth                  = growth_model, 
                    changepoint_prior_scale = cps, 
                    yearly_seasonality      = True, 
                    weekly_seasonality      = True, 
                    holidays                = us_holidays )
        m.fit(pdf)


        future                              = m.make_future_dataframe (periods=duration)
        if not (growth_model == "linear"):
            future["cap"]                   = cap
        forecast                            = m.predict(future)

        print("Completed forecasting for grouped agg, duration: {}, in {}secs with {} growth model".format(duration, time.time()-start_time, growth_model))


        if to_log:
            forecast["yhat"] = np.exp(forecast["yhat"])

        if not yhat_lower:
            forecast_total = forecast[(forecast["ds"] >= end_ts) & (forecast["ds"] <= forecast_end) ]["yhat"].sum()
        else:
            forecast_total = forecast[(forecast["ds"] >= end_ts) & (forecast["ds"] <= forecast_end) ]["yhat_lower"].sum()
    else:
        # when the forecast total is specified for this group;
        # just take the value and attribute
        m        = None
        forecast = None
        forecast_total = fcast_total
        

        
    if test:
        acc_df = tz_df[(tz_df['date'] >= end_ts) & (tz_df['date'] <= forecast_end)].groupby("zip").sum()
        acc_df = acc_df.join(tz_ds["percent"])
    else:
        acc_df = tz_ds[["percent"]]
    acc_df["forecast"] = forecast_total * acc_df["percent"]
    if test:
        acc_df["accuracy"] = (acc_df["cobroke_total"] - acc_df["forecast"]) * 100 / acc_df["cobroke_total"]
        print ("Aggregated forecast total {}, orig: {}, err: {}%".format(forecast_total, acc_df["cobroke_total"].sum(), round((acc_df["cobroke_total"].sum() - forecast_total) * 100 / acc_df["cobroke_total"].sum()), 2))
    
    return m, forecast,acc_df


def runGroups():
    rate = pd.read_csv(os.path.join(input_dir,"rated_zips.csv"))
    # filter the zip codes that have more than 14 leads in the first 6 months of 2016..
    # the limit is any zip more than 24 for an year
    
    rate = rate[rate["16_total"] >= 14]

    # Total leads divided into different buckets
    buckets = [(14,29),(30, 50), (51,70), (71,100),(101,140),(141,200), (201,275), (276,400),(401, 3000)]

    # Rate of change divided into 10 different buckets
    rate_buckets = [
        ( 2.0,  6.0, 0.03, 1.8),
        ( 1.5,  2.0, 0.03, 1.8),
        ( 1.0,  1.5, 0.03, 1.8),
        ( 0.0,  1.0, 0.04, 1.8),
        (-0.2,    0, 0.05, 1.8),
        (-0.4, -0.2,  0.4, 2),
        (-0.6, -0.4,    1, 2),
        (-0.8, -0.6,    2, 4),
        (-0.9, -0.8,    2, 4),
        (-1.0, -0.9,  0.5, 5)
    ]

    summary = pd.DataFrame()
    summary_name = strftime("%Hh%Mm%Ss %Y-%m-%d.csv", gmtime())

    group_df = {}
    
    for b in buckets:
        df1 = rate[(rate["16_total"] >= b[0]) & (rate["16_total"] <= b[1])]
               
        for rb in rate_buckets:
            xx = None

            xx      = df1[(df1["rate"] >= rb[0]) & (df1["rate"] < rb[1])]
            if len (xx) == 0:
                continue

            if rb[0] > 1:
                yhat_lower = True
            else:
                yhat_lower = False
                
            m,f,a   = predictByGroup(filter_list= list(xx.zip.values), cps=rb[2], cap_per=rb[3], yhat_lower=yhat_lower)
            eighty  = len(a[(a["accuracy"] > -20) & (a["accuracy"] <= 20)])
            seventy = len(a[(a["accuracy"] > -30) & (a["accuracy"] <= 30)])

            s = "{}: ({} : {}) 80% - {}, 70% - {} of {}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), rb[0], rb[1], eighty, seventy, len(a))
            print(s)
            with open(os.path.join(input_dir,"results_group.txt"), "a") as f:
                f.write(s)
    
            ax             =  a[["cobroke_total", "forecast", "percent", "accuracy"]].rename(columns={"cobroke_total":"2016Jan2JunActual", "accuracy":"error_per", "forecast":"2016Jan2JunForecast", "percent": "percent"})

            group_id       = random.randint(10,10000)
            ax["group_id"] = group_id

            with open(os.path.join(output_dir,summary_name,"group_info.json"), "a") as f1:
                f1.write(json.dumps({
                    "group_id": group_id,
                    "rate_buckets" : rb,
                    "buckets" : b
                }))
            # forecast for the group
            m,f,a   = predictByGroup(filter_list= list(xx.zip.values), cps=0.001, cap_per=1.2, test=False)

            m.plot(f)
            plt.savefig(os.path.join(output_dir,str(group_id)+".png"))
            
            ay      = a[["forecast"]].rename(columns={"forecast":"52WeekForecast"})

            ax      = ax.join(ay)
            summary = summary.append(ax)

    print ("Created forecast summary at {}".format(summary_name))
    summary.to_csv(os.path.join(output_dir,summary_name))

def forecastGroup(summary_file):
      
    rdf = pd.read_csv(os.path.join(output_dir, summary_file))
    fcast_totals = {}                      
    # for each group
    # get the forecast_total
    # compute total percentage

    m, f, a = predictAgg (end="2016--6-30", duration=365, cps=0.001)

    # take the aggregated totoal for the 52 weeks
    forecast_agg_total = a[(a["date"] >= "2016-07-01") & (a["date"] < "2017-07-01")]["yhat_e"].sum()
    print("TOTAL:{}".format(forecast_agg_total))
    
    for group in rdf["group_id"].unique():
        gdf = rdf[rdf["group_id"] == group]
        fcast_totals[group] = gdf["2016Jan2JunActual"].sum()

    fdf = pd.DataFrame.from_dict(fcast_totals, orient="index")
    fdf.columns = ["ftotal"]
    fdf["percent"] = fdf["ftotal"] / fdf["ftotal"].sum()
    fdf["52WeekForecastGrouped"] = np.round(forecast_agg_total * fdf["percent"])

    forecast_df = pd.DataFrame()
    
    for group in rdf["group_id"].unique():
        gdf = rdf[rdf["group_id"] == group]
        # read the row from fdf for this group id
        gdf ["52WeekForecastGrouped"] = gdf["percent"] * fdf.ix[group]["52WeekForecastGrouped"]
        forecast_df.append(gdf)

    forecast_df.to_csv(os.path.join(output_dir, "result-"+strftime("data/%Hh%Mm%Ss-%m-%d.csv", gmtime())))
                       
        
    return fdf, forecast_df
    
        


def computeMonthly():
    # for all the zip codes considered earlier..> 14 for 2016Jan - 2016Jun
    tz_df = df[(df['date'] >= "2016-01-01") & (df['date'] <= "2016-06-30")].groupby("zip").sum()
    tz_df = tz_df[tz_df["cobroke_total"] >= 14]
    tz_df = tz_df.reindex()
    
    # compute the total
    xx = df[(df["date"] >= "2015-07-01") & (df["date"] < "2016-07-01")].groupby("zip").sum() ["cobroke_total"]
    xx = xx[xx.index.isin(list(tz_df.index.values))]
    xx = xx.to_frame()
    xx = xx.rename(columns={"cobroke_total":"total"})
    
    # compute monthly aggregates    
    xy = df[(df["date"] >= "2015-07-01") & (df["date"] < "2016-07-01")].set_index("date")
    xy.index = pd.to_datetime(xy.index)
    xy = xy.groupby(["zip",pd.TimeGrouper("M")]).sum()
    # unstack the multilevel index so that it gets easier to filter and join for the zips we want later on
    xy = xy.unstack().fillna(0)

    # combine
    zips_seasonality = xx.join(xy)
    # rename the monthly columns...col names from unstack makes it harder to read
    col_names = ["2015-07", "2015-08", "2015-09", "2015-10","2015-11","2015-12","2016-01","2016-02","2016-03","2016-04","2016-05","2016-06"]
    for i in range(1,13):
        zips_seasonality.columns.values[i] = col_names[i-1]
        
    # read the 52 week forecasted values for each zip and join them with the past monthly data
    x1 = pd.read_csv(os.path.join(output_dir,"result.csv"))
    x2 = x1[["zip","52WeekForecastGrouped"]].set_index("zip").join(zips_seasonality)
    x2["52WeekForecastGrouped"] = np.round(x2["52WeekForecastGrouped"])
    # compute %ages & get the forecasted results monthly for each zip
    # attribute for each zip monthly
    for i in col_names:
        x2[i] = np.round((x2[i] / x2["total"]) * x2["52WeekForecastGrouped"])
    x2["Forecast_Total"] = x2[col_names].sum(axis=1)
    col_names_future =     ['2016-07',
                            '2016-08',
                            '2016-09',
                            '2016-10',
                            '2016-11',
                            '2016-12',
                            '2017-01',
                            '2017-02',
                            '2017-03',
                            '2017-04',
                            '2017-05',
                            '2017-06']
    del x2["52WeekForecastGrouped"]
    del x2["total"]
    for i in range(0,12):
        x2.columns.values[i] = col_names_future[i]
    # store it as results_monthly.csv
    x2.to_csv(os.path.join(output_dir, "result_monthly.csv"))
    print("Completed forecasting for monthly, duration 2016-07 to 2017-06.Output csv files{}".format((output_dir, "result_monthly.csv")))
    return

                       
if __name__ == "__main__":

    #plac.call(predictTop)
    plac.call(runGroups)
    #plac.call(trainZip)