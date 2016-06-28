---
layout: post
title:  "Forecasting with machine learning models"
date:   2016-06-27 21:33:43 +0100
categories: jekyll update
---

Recently I had to work on a project where I had to compute long-term forecasts for a large number of time series. Coming from a statistics background I was familiar with some of the standard methods in the literature, like seasonal [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) models and [state-space models](https://en.wikipedia.org/wiki/State-space_representation). I like Bayesian dynamic linear models (DLMs) in particular because of their flexibility and ease of interpretation (check out [this](http://multithreaded.stitchfix.com/blog/2016/04/21/forget-arima/) post over at Stitch Fix for an excellent discussion of these models)

### Taking a machine learning point of view

For me, there were two main obstacles that prevented me from using these models:

1. I needed **longer-term** forecasts for each time series: ARIMA models are typically well-suited for short-term forecasts, but not for longer term forecasts due to the autoregressive part of the model. 
2. I had to provide forecasts for a very **large number** of time series: Hence, I needed a computationally cheap type of model. This ruled out using some of the Bayesian models because the MCMC algorithms are not computationally cheap.

Instead I opted for a more algorithmic point of view and decided to try out some (cheap) machine learning methods. Most of these methods are designed for independent and identically distributed (IID) data however, so it is interesting to see how we can apply these models to non-IID data (e.g. time series data). 

### Forecasting strategies

In this blog post we will make the following *non-linear autogressive representation* (NAR) assumption. Let $$y_t$$ denote the value of the time series at time point $$t$$, then we assume that 

$$ y_t = f(y_{t - 1}, \ldots, y_{t - n})+ \epsilon_t, $$

for some autoregressive order $$n$$ and where $$\epsilon_t$$ represents some noise at time $$t$$ and $$f$$ is an arbitrary and unknown function. The goal is to learn this function $$f$$ from the data and obtain forecasts for $$t + h$$, where $$h \in \{1, \ldots, H\}$$. Hence we are interested in predicting the next $$H$$ data points, not just the $$H$$-th data point, given the history of the time series. 

When $$H = 1$$ (*one-step ahead forecasting*), it is straightforward to apply most machine learning methods on your data. In the case where we want to predict multiple time periods ahead ($$H > 1$$) things become a little more interesting.

In this case there are three common ways of forecasting:

* Iterated one-step ahead forecasting.
* Direct $$H$$-step ahead forecasting.
* Multiple input multiple output models.

### Iterated forecasting

In iterated forecasting, we optimize a model based on a one-step ahead criterion and when calculating a $$H$$-step ahead forecast, we iteratively feed the forecasts of the model back in as input for the next prediction. In Python, a function that would compute the iterated forecast could look like this:


```python
def generate_features(x, forecast, window):
    augmented_time_series = np.hstack((x, forecast))

    return augmented_time_series[-window:].reshape(1, -1)

def iterative_forecast(model, x, window, H):
    """ Implements iterative forecasting strategy
    
    Arguments:
    ----------
        model: scikit-learn model that implements a predict() method and is trained on some data x
        x: history of the time series
        h: number of time periods needed for the h-step ahead forecast
    """
    forecast = np.zeros(H)    
    forecast[0] = model.predict(x.reshape(1, -1))
    
    for h in range(1, H):
        features = generate_features(x, forecast[:h], window)

        forecast[h] = model.predict(features)
    
    return forecast
```

To understand the disadvantage of this method a bit better, it helps to go back to the original goal of our problem. What we are really trying to do is to approximate $$\mathbb{P}\left[ Y \| X \right]$$ where $$Y \in \mathbb{R}^H$$ and $$X \in \mathbb{R}^n$$. We can visualise this distribution by using a graphical model. In the case $$n = 2$$ the distribution of the time series data can be represented as follows

![]({{ BASE_PATH }}/images/2016_06_30/ts-graphical-model.jpg)

Now, the distribution of our approximation is actually a bit different and looks more like this:

![]({{ BASE_PATH }}/images/2016_06_30/ts-iterated-prediction.jpg)

The iterated strategy returns an unbiased estimator of $$\mathbb{P}\left[Y \| X\right]$$ since it preserves the stochastic dependencies of the underlying data. In terms of the bias-variance trade-off however, it suffers from high variance due to the accumulation of error in the individual forecasts. This means we will get a low performance over longer time horizons $$H$$.

### $$H$$-step ahead forecasting

In direct $$H$$-step ahead forecasting, we learn $$H$$ different models of the form

$$ y_{t + h} = f_h (y_t, \ldots, y_{t - n}) + \epsilon_{t + h}, $$

where $$h \in \{1, \dots, H\}$$, $$n$$ is the number of past data points and $$f_h$$ is any arbitrary learner. Training the models $$f_h$$ in Python is relatively straightforward, you only need to use different (lagged) versions of your training data $$X$$ and response $$y$$. 


```python
def ts_to_training(x, window, h):
    """ Generates a training and test set from a time series assuming we want to calculate a h-step ahead forecast
    
    Arguments:
    ----------
        x: Numpy array that contains the time series.
        h: Number of periods to forecast into the future.
    """
    n = x.shape[0]
    nobs = n - h - window + 1
    
    features = np.zeros((nobs, window))
    response = np.zeros((nobs, 1))
    
    for t in range(nobs):
        features[t, :] = x[t:(t + window)]
        response[t, :] = x[t + window]
        
    return features, response


def direct_forecast(model, x, window, H):
    """ Implements direct forecasting strategy
    
    Arguments:
    ----------
        model: scikit-learn model that implements fit(X, y) and predict(X)
        x: history of the time series
        H: number of time periods needed for the H-step ahead forecast
    """
    forecast = np.zeros(H)

    for h in range(1, H + 1):
        X, y = ts_to_training(x, window=window, h=h)
        
        fitted_model = model.fit(X, y)
        
        forecast[h - 1] = fitted_model.predict(X[-1, :].reshape(1, -1))
    
    return forecast
```

We can again visualise the distribution this strategy approximates in a graphical model:

![]({{ BASE_PATH }}/images/2016_06_30/ts-direct-prediction.jpg)

Here we see that this approach does not suffer from the accumulation of error, since each model $$f_h$$ is tailored to predict horizon $$h$$. However, since the models are trained independently no statistical dependencies between the predicted values $$y_{t + h}$$ are guaranteed. This strategy is also computationally heavy because we now need to train $$H$$ independent models.

### Multiple input multiple output models

Finally, we can also train one model that takes multiple inputs and returns multiple outputs:

$$ [y_{t + H}, \ldots, y_{t  +1}] = f(y_t, \ldots, y_{t - n}) + \textbf{w}. $$

The forecasts are provided in one step, and any learner $$f$$ that can deal with a multi-dimensional response can be used. This means that you only have to take care when you construct your feature and response data sets. In Python, this could look like this.


```python
def ts_to_mimo(x, window, h):
    """ Transforms time series to a format that is suitable for multi-input multi-output regression models.
    
    Arguments:
    ----------
        x: Numpy array that contains the time series.
        window: Number of observations of the time series that will form one sample for the multi-input multi-output model.
        h: Number of periods to forecast into the future.
    """
    n = x.shape[0]
    nobs = n - h - window + 1
    
    features = np.zeros((nobs, window))
    response = np.zeros((nobs, h))
    
    for t in range(nobs):
        features[t, :] = x[t:(t + window)]
        response[t, :] = x[(t + window):(t + window + h)]
        
    return features, response
```

The results of the above function can then be piped into any model that takes multi-dimensional input and output data. In long term prediction scenarios, the recursive and direct strategies neglect stochastic dependencies between future values. Using models that take multi-dimensional inputs and outputs therefore seems the most natural choice of models as the main advantages of the MIMO forecasting strategy are that 

1. Only one model is trained instead of $$H$$ different models.
2. No conditional independence assumptions are made (c.f. direct strategy).
3. There is no accumulation of error of individual forecasts (c.f. iterated strategy).

### Using $$k$$-nearest neighbours to predict monthly car sales

To illustrate the methods discussed above we'll go through a quick example where whe use a $$k$$-nearest neighbour model as the learner. We'll use a data set of monthly car sales in Quebec starting from January 1960 to December 1968 which I obtained from [DataMarket](https://datamarket.com/data/set/22n4/monthly-car-sales-in-quebec-1960-1968#!ds=22n4&display=line).

First we'll define some helper functions and import the necessary packages.


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV
```


```python
def ts_to_training(x, window, h):
    """ Generates a training and test set from a time series assuming we want to calculate a h-step ahead forecast
    
    Arguments:
    ----------
        x: Numpy array that contains the time series.
        h: Number of periods to forecast into the future.
    """
    n = x.shape[0]
    nobs = n - h - window + 1
    
    features = np.zeros((nobs, window))
    response = np.zeros((nobs, 1))
    
    for t in range(nobs):
        features[t, :] = x[t:(t + window)]
        response[t, :] = x[t + window]
        
    return features, response

def create_cv_split(X, perc=0.75):
    n = X.shape[0]
    n_train = np.floor(perc * n).astype(int)
    split = np.ones(n)
    split[:n_train] = -1

    return PredefinedSplit(split)

def generate_features(x, forecast, window):
    augmented_time_series = np.hstack((x, forecast))

    return augmented_time_series[-window:].reshape(1, -1)

def iterative_forecast(model, x, window, H):
    """ Implements iterative forecasting strategy
    
    Arguments:
    ----------
        model: scikit-learn model that implements a predict() method and is trained on some data x
        x: history of the time series
        h: number of time periods needed for the h-step ahead forecast
    """
    forecast = np.zeros(H)    
    forecast[0] = model.predict(x.reshape(1, -1))
    
    for h in range(1, H):
        features = generate_features(x, forecast[:h], window)

        forecast[h] = model.predict(features)
    
    return forecast
  
def direct_forecast(model, x, window, H):
    """ Implements direct forecasting strategy
    
    Arguments:
    ----------
        model: scikit-learn model that implements fit(X, y) and predict(X)
        x: history of the time series
        H: number of time periods needed for the H-step ahead forecast
    """
    forecast = np.zeros(H)

    for h in range(1, H + 1):
        X, y = ts_to_training(x, window=window, h=h)
        
        fitted_model = model.fit(X, y)
        
        forecast[h - 1] = fitted_model.predict(X[-1, :].reshape(1, -1))
    
    return forecast

def time_series_to_mimo(x, window, h):
    """ Transforms time series to a format that is suitable for multi-input multi-output regression models.
    
    Arguments:
    ----------
        x: Numpy array that contains the time series.
        window: Number of observations of the time series that will form one sample for the multi-input multi-output model.
        h: Number of periods to forecast into the future.
    """
    n = x.shape[0]
    nobs = n - h - window + 1
    
    features = np.zeros((nobs, window))
    response = np.zeros((nobs, h))
    
    for t in range(nobs):
        features[t, :] = x[t:(t + window)]
        response[t, :] = x[(t + window):(t + window + h)]
        
    return features, response

def plot_predictions(y_history, y_hat, y_true, plot_obs=False):
    history_idx = range(y_history.shape[0])
    pred_idx = range(y_history.shape[0], y_history.shape[0] + y_hat.shape[0])
    
    plt.figure()
    plt.plot(history_idx, y_history, hold=True, label="Observed")
    plt.plot(pred_idx, y_hat, hold=True, label="Forecast")
    
    if plot_obs:
        plt.plot(pred_idx, y_true, hold = True, label = "Observed")
        
    plt.title("Observed versus forecasted observations")
    plt.legend(loc = "best")
    plt.grid(True)
    plt.show()
```

The time series looks like it has a relatively well behaved seasonal pattern and an upwards trend.

```python
ts = np.genfromtxt("monthly-car-sales-in-quebec-1960.csv", delimiter=",")[1:, 1:].reshape(-1)

plt.plot(ts)
```

![]({{ BASE_PATH }}/images/2016_06_30/car-sales.png)

```python
# Multi-input multi-output model
# Set up some global parameters and the model
c = 80
H = 20
window = 20

model = KNeighborsRegressor(algorithm="auto")
param_grid = {"n_neighbors": [2 ** x for x in range(5)],
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}

X, y = time_series_to_mimo(x=ts[:c], window=window, h=H)

cv_split = create_cv_split(X, perc=0.75)
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring="mean_squared_error",
                           cv=cv_split,
                           refit=True)

grid_search.fit(X, y)
```

```python
# Now forecast the test period
X_test = ts[(c - window):c].reshape(1, -1)
y_test = ts[c:(c + H)]

y_hat = grid_search.predict(X_test).reshape(-1)

plot_predictions(ts[:c], y_hat, y_test, plot_obs=True)
```

![]({{ BASE_PATH }}/images/2016_06_30/forecast-mimo.png)

## Iterative forecast

```python
c = 80
H = 20
window = 5

model = KNeighborsRegressor(algorithm="auto")
param_grid = {"n_neighbors": [2 ** x for x in range(5)],
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}

X, y = ts_to_training(x=ts[:c], window=window, h=1)

cv_split = create_cv_split(X, perc=0.75)
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring="mean_squared_error",
                           cv=cv_split,
                           verbose=True,
                           refit=True)

grid_search.fit(X, y)

# Now forecast the test period
X_test = ts[(c - window):c]
y_test = ts[c:(c + H)]
y_hat = iterative_forecast(grid_search, X_test, window, H)

plot_predictions(ts[:c], y_hat, y_test, plot_obs=True)
```

![]({{ BASE_PATH }}/images/2016_06_30/forecast-iterated.png)

## Direct strategy

```python
# We use the same model from the iterated strategy
# Now forecast the test period
X_test = ts[(c - window):c]
y_test = ts[c:(c + H)]

model_params = {'n_neighbors': 2, 'weights': 'distance', 'metric': 'manhattan'}
model = KNeighborsRegressor(**model_params)
y_hat = direct_forecast(model, ts, window, H)

plot_predictions(ts[:c], y_hat, y_test, plot_obs=True)
```

![]({{ BASE_PATH }}/images/2016_06_30/forecast-direct.png)



