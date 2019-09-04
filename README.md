# ProductDemandForecast
Two different scripts for modeling the Corporación Favorita dataset:
### Top3ProductsMonthlyForecast.R

The sales of three products that are among the most sold are modeled using linear regression, ridge regression and XGBoost, and monthly predictions are made and plotted.

Variable importances are also plotted for the three products using these three different models.

Interestingly the linear and XGBoost models use lagged sales as their most important predictors, but the ridge models do not.

![pred_vs_act](https://github.com/KaroRonty/ProductDemandForecast/blob/master/pred_vs_actual.png)
![var_imp](https://github.com/KaroRonty/ProductDemandForecast/blob/master/variable_importances.png)

### ProductDemandForecast.r
Sales forecasting done on each product-store pair using lasso regression

Includes a function for plotting predictions and actuals. Results of the models are below.

![Pred_vs_act](https://github.com/KaroRonty/ProductDemandForecast/blob/master/predictions.png)
![R-squared](https://github.com/KaroRonty/ProductDemandForecast/blob/master/r-squared.png)
![P-values](https://github.com/KaroRonty/ProductDemandForecast/blob/master/p-values.png)
