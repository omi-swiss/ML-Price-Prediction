# ML-Price-Prediction
The project focuses on forecasting the prices of Avocado in the US. We look into multiple approaches using Machine Learning to see what is an optimal model to predict the prices. 

## Aim
Predict Avocado prices in the US.

## Data
This dataset comprises weekly retail scan data covering National retail volume (units) and prices of Hass avocados. It spans from April 2015 to March 2018. The "AveragePrice" column represents the cost per avocado, even for multiple avocados sold in bags. The dataset focuses exclusively on Hass avocados and excludes other varieties. Key columns include:

- **Date**: Date of the observation.
- **AveragePrice**: Average price of a single avocado.
- **Type**: Conventional or organic.
- **Region**: Region of the observation.
- **Total Volume**: Total avocados sold.
- **4046**: Total avocados with PLU 4046 sold.
- **4225**: Total avocados with PLU 4225 sold.
- **4770**: Total avocados with PLU 4770 sold.
- **Total Bags**: Total bags sold.
- **Small/Large/XLarge Bags**: Total bags sold by size.

The dataset covers two avocado types and multiple regions, enabling diverse analyses. Our focus will be on the entire dataset.

## Exploratory Analysis
In this section, we conduct an exploratory analysis of the dataset to gain insights into avocado pricing trends, seasonal patterns, and regional variations.

## Feature Engineering
Here, we perform feature engineering to create relevant features that can enhance the performance of our price prediction models.

## Modeling
We employ two time series forecasting models in this section:

### ARIMA Model
The Autoregressive Integrated Moving Average (ARIMA) model is used for time series forecasting of avocado prices. It leverages historical price data to make predictions.

### SARIMA Model
The Seasonal Autoregressive Integrated Moving Average (SARIMA) model extends ARIMA by considering seasonal patterns in avocado prices. It provides more accurate forecasts by accounting for seasonality.

