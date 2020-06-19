# Data-Mining-Assignment
## Title: Property Stock Prediction with Historical Data, Bursa Announcement Data and Quarterly Report Data

## Team member:
1) WQD180124 Choo Jian Wei

## Milestone 1: Acquisition of Data
Requirements for python script (Libraries & Driver):
1) pandas
2) Beautiful soup
3) request
4) selenium
5) pandas_datareader
6) time
7) Chromedriver (make sure it's the same version with your google chrome)

There are 5 parts:
1) Scrape data from MalysianStockBiz to get Top 10 Property Stock data sort by Market Cap
2) Top 10 Property Stock Historical Data (10 Years)
3) KLSE data from Yahoo Finance (10 Years)
4) Bursa Announcement data (Max)
5) Quarterly report data from MalaysianStockBiz (Max)

Link to youtube video: https://youtu.be/LAiL65JRsaU


## Milestone 2: Management of Data
Requirements for python script (Libraries & Driver):
1) pandas
2) VMWare https://www.vmware.com/products/workstation-player/workstation-player-evaluation.html
3) Hive & Hadoop using Dr. Hoo image https://drive.google.com/file/d/11WrsLOXzlveWJ2TnX7wKRpDCcH-03YWY/view?usp=sharing 

There are 5 parts:
1) Data cleaning
2) Checking the table columns and their respective data type
3) Creating Hive tables
4) Using VMWare to run Hive
5) Loading data into Hive

Link to youtube video: https://youtu.be/UEbUq-h-ksA

## Week 7 Data Cleaning
Requirements for python libraries:
1) Numpy
2) Pandas

Data cleaning for 5 datasets:
1) Change column type
2) Check for null and fix null values
3) Convert data format of certain columns
4) Rename column name


## Milestone 3: (Accessing data warehouse or data lake using Python - Big Query)
Requirements:
1) pandas
2) Google Cloud Platform account (Free $300 Credit)

There are 3 parts:
1) Converting data into CSV format as GCP doesn't allow "|" as separator
2) Creating a project in GCP
3) Uploading the data too GCP bucket
4) Create table for each of the files
5) Linking up Big Query with Python
5a) Authentication can be learn via: https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries#client-libraries-install-python
5b) Add downloaded JSON file to your bashrc and source it. (E.g. export GOOGLE_APPLICATION_CREDENTIALS="/home/user/Downloads/wqd_datamining) 

Link to youtube video: https://www.youtube.com/watch?v=U67gD8FFybw

## Milestone 4: Interpretation of data & Communication of insights
There are a few parts:
a) Top 10 property stocks analysis 
b) Top 10 property stocks vs KLSE index (Correlation)
c) Alpha & Beta Analysis (Property stocks compared to KLSE market index)
d) Prediction using LSTM, Linear Regression and Support vector regressor
Link to youtube video: https://youtu.be/2EcMVJUbSL8
Analytics: We should invest in 5606.KL, 1651.KL and 5200.KL (Negative Beta Stocks)

## Milestone 5: Deployment using Flask
a) Create the run.py files and importing the relevant libraries such as flask, numpy, pandas, plotly, tensorflow & keras.
b) Create the html files for each pages and link them to the url in run.py
c) Plot using plotly and figures are shown using img tag in html
d) Train model for each stocks and load it to the python file.
e) Beautify the website using Bootstrap classes.