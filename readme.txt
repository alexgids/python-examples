 Leads Forecast:
----------------

This readme.txt outlines the usage of the forecast framework. Below sections contains the steps for build installation.


Contents of package:
--------------------
The content of the packages 

- forecast-0.1.tar.gz
- readme.txt


Requirements: 
-------------
Python version < 3.5
Anaconda 3 (recommended)

Python Packages required:
-------------------------
- fbprophet
- pandas
- numpy
- matplotlib
- plac

Quick Start Guide
-----------------

1.  Extract the package forecast-0.1.tar.gz and 
2.  forecast-0.1.tar.gz contains forecast.py which has all the methods for using the framework.
3.  data and outputs contains the csv files used by the frame work.
4.  Place the script forecast.py and the folders data and outputs in the same path.
5.  Run the python shell (ipython recommended) and from the same path and import forecast. 
    Eg : import forecast as f 
         f.predictAgg()
6. The following are the methods and their descriptions 

predictAgg(): Function for generating the aggregated forecast for 52 weeks.

arguments: accepts no arguments.

eg: f.predictAgg()

predictZip(): Predicts the leads forecast for the give zip code

arguments:  zipcode 

eg: f.predictZip("33029")

predictTop(): Predicts top n zip codes based on no of leads

arguments: n where n =1,2,3...

eg: f.predictTop(10)

predict
Important Links:
----------------

fbprophet - https://facebook.github.io/prophet/

Aggregated Forecast for 52 weeks (July 16 - Jun 17) & Monthly Forecast:
