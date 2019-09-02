library(data.table) # fread function
library(zoo) # approximating NAs
library(tidyr) # unite function
library(padr) # padding
library(dtplyr) # converting dplyr to data.table
library(dplyr) # data wrangling
library(lubridate) # handling dates
library(purrr) # handling nested data
library(glmnet) # lasso regression
library(caret) # XGBoost model, varImp function
library(tidyr) # replacing NAs
library(ggplot2) # plotting
library(gridExtra) # plotting multiple plots
library(tibble) # add_column function
library(multidplr) # parallel dplyr

# Reading & transforming the data ----

# Oil price data
# Aggregate oil price to monthly and approximate NAs
oil_df <- fread("oil.csv") %>%
  as_tibble() %>% 
  mutate(year = year(date),
         month = month(date))%>% 
  group_by(year, month) %>% 
  # Select the first oil price of each month
  summarise(oilprice = first(dcoilwtico)) %>% 
  na.approx() %>%  
  as_tibble() %>% 
  mutate(month = as.character(month))

# Sales data
# Select interval and aggregate to monthly
data5 <- fread("train.csv") %>% 
  lazy_dt() %>% 
  filter(date >= as.Date(last(date)) - years(4)) %>% 
  mutate(date = as.Date(date),
         year = year(date),
         month = month(date) %>% as.character(),
         promo = replace_na(onpromotion, 0),
         store_nbr = as.character(store_nbr)) %>% 
  group_by(year, month, item_nbr, store_nbr) %>% 
  summarise(sales = sum(unit_sales),
            promo = mean(promo)) %>% 
  as_tibble()

# TODO: change
data5 <- readRDS("data5.rds")
bu <- data5

# Find out most sold products
most_sold <- data5 %>% 
  group_by(item_nbr) %>% 
  summarise(sales = sum(sales)) %>% 
  arrange(-sales) %>% 
  head(6)

# Select only the most sold products
data5 <- data5 %>% 
  filter(item_nbr %in% most_sold$item_nbr) %>% 
  mutate(year_month = as.Date(paste0(year, "-", month, "-01")))

# Use ~67/33 training/test split
split_date <- last(data5$year_month) - years(1)
# TODO: remove
# Use two of the three years as a training set
# training_set_size <- 3 / 4

# TODO: remove/keep padding
# Pad months with zero sales
data5 <- data5 %>% 
  group_by(store_nbr, item_nbr) %>% 
  pad(interval = "month", break_above = 1e9)

# Add back dates and replace padded NAs with zeros
data5 <- data5 %>% 
  lazy_dt() %>% 
  mutate(year = year(year_month),
         month = month(year_month) %>% as.character(),
         sales = replace_na(sales, 0),
         promo = replace_na(promo, 0)) %>% 
  as_tibble()# %>% 
  #select(-temp_date)

# Combine the sales data with the oil price data
full_data <- data5 %>% 
  left_join(oil_df)

# Make lagged sales variables
temp2 <- full_data %>%  # temp for testing
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(store_nbr, item_nbr) %>% 
  mutate(sales_lag12 = lag(sales, 12),
         sales_lag1 = lag(sales, 1)) %>% 
  na.omit()

# Splitting into training and test sets ----
# Make test set with dates and actual sales
to_model <- temp2 %>% 
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(item_nbr) %>% 
  filter(year_month <= split_date) %>% 
  summarise(year_month_train = list(year_month),
            sales_train = list(sales),
            stores_train = list(store_nbr))

# Make test set with dates and actual sales
to_model <- temp2 %>% 
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(item_nbr) %>% 
  filter(year_month > split_date) %>% 
  summarise(year_month_test = list(year_month),
            sales_test = list(sales),
            stores_test = list(store_nbr)) %>% 
  inner_join(to_model, .)

# TODO
# to_model

# Split into training and test sets by date
train <- temp2 %>% 
  filter(year_month <= split_date)
  
test <- temp2 %>% 
  filter(year_month > split_date)

# Do model matrices for training data (dummy variables etc.)
model_data <- train %>% 
  group_by(item_nbr) %>% 
  do(training = safely(model.matrix)(sales ~
                                       year +
                                       factor(month, levels = 1:12) +
                                       promo +
                                       oilprice +
                                       sales_lag12 +
                                       sales_lag1,
                                     data = .)$result[, -1]) %>%
  inner_join(to_model, .)

# Do model matrices for training data (dummy variables etc.)
model_data <- test %>% 
  group_by(item_nbr) %>% 
  do(test = safely(model.matrix)(sales ~
                                   year +
                                   factor(month, levels = 1:12) +
                                   promo +
                                   oilprice +
                                   sales_lag12 +
                                   sales_lag1,
                                 data = .)$result[, -1]) %>%
  inner_join(model_data, .)

# Modeling ----
# Do cross validations to extract lamdas for the models using training set
model_data <- model_data %>% 
  group_by(item_nbr) %>% 
  do(cv = safely(cv.glmnet)(pluck(.$training, 1),
                            pluck(.$sales_train, 1),
                            alpha = 0)$result) %>% 
inner_join(model_data, .)

# Obtain just the lambdas
model_data <- model_data %>% 
  group_by(item_nbr) %>% 
  add_column(lambda = as.numeric(as.character(lapply(.$cv, `[[`, 9)))) %>% 
  # Replace possible null lambdas with NAs
  mutate(lambda = modify_if(lambda, is.null, ~ NA)) %>% 
  select(-cv)

# Make the models ----
# Linear
linear_models <- model_data %>% 
  group_by(item_nbr) %>% 
  do(model = safely(lm)(sales ~ .,
                        data = cbind(sales = pluck(.$sales_train, 1),
                                     pluck(.$training, 1)) %>%
                          as_tibble())$result)

# Ridge
ridge_models <- model_data %>% 
  group_by(item_nbr) %>% 
  do(model = safely(glmnet)(pluck(.$training, 1),
                            pluck(.$sales_train, 1),
                            lambda = pluck(.$lambda, 1),
                            alpha = 0)$result)

# XGboost with grid search
# Create cluster for parallel computing
cluster <- new_cluster(2)
cluster %>%
  cluster_library("purrr") %>%
  cluster_library("caret")

time <- Sys.time() # 32 min
xgb_models <- model_data %>% 
  partition(cluster) %>%
  group_by(item_nbr) %>% 
  do(model = safely(train)(y = pluck(.$sales_train, 1),
                           x = pluck(.$training, 1),
                           method = "xgbTree",
                           metric = "RMSE",
                           trControl = trainControl(method = "repeatedcv"),
                           tuneGrid = expand.grid(
                             # number of trees, higher if size of data is high
                             nrounds = c(5, 10, 15, 20),
                             # smaller value prevents overfitting, 6; 0-inf
                             max_depth = c(6, 10, 15, 25),
                             # smaller value prevents overfitting, 6; 0-inf
                             eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
                             # higher value = more conservative, 0; 0-inf
                             gamma = c(0, 5),
                             # 1; 0-1
                             colsample_bytree = c(0.1, 0.3, 0.5, 0.8, 1),
                             # higher value = more conservative, 1; 0-inf
                             min_child_weight = 1,
                             # smaller value prevents overfitting, 1; 0-1,
                             subsample = c(0.5, 1)),
                           allowParallel = TRUE)$result) %>% 
  collect()
(time <- Sys.time() - time)

# Model evaluation ----
# Function to make predictions and calculate accuracy measures for both sets
make_predictions <- function(model_df, lambda = FALSE){
  # Make predictions using the models (training set)
  if(lambda){
  result_data <- model_df %>% 
    inner_join(model_data, by = "item_nbr") %>% 
    group_by(item_nbr) %>% 
    do(predictions_train = safely(predict)(pluck(.$model, 1),
                                             s = .$lambda,
                                           newx = pluck(.$training, 1)
                                           )$result) %>% 
    inner_join(model_data, ., by = "item_nbr")
  
  # Make predictions using the models (test set)
  result_data <- model_df %>% 
    inner_join(model_data, by = "item_nbr") %>% 
    group_by(item_nbr) %>% 
    do(predictions_test = safely(predict)(pluck(.$model, 1),
                                            s = .$lambda,
                                          newx = pluck(.$test, 1))$result) %>% 
    inner_join(result_data, ., by = "item_nbr")
  } else {
    result_data <- model_df %>% 
      inner_join(model_data, by = "item_nbr") %>% 
      group_by(item_nbr) %>% 
      do(predictions_train = safely(predict)(pluck(.$model, 1),
                                             newdata = pluck(.$training, 1) %>% 
                                               as_tibble()
      )$result) %>% 
      inner_join(model_data, ., by = "item_nbr")
    
    # Make predictions using the models (test set)
    result_data <- model_df %>% 
      inner_join(model_data, by = "item_nbr") %>% 
      group_by(item_nbr) %>% 
      do(predictions_test = safely(predict)(pluck(.$model, 1),
                                            newdata = pluck(.$test, 1) %>% 
                                              as_tibble())$result) %>% 
      inner_join(result_data, ., by = "item_nbr")
  }
  
  # Evaluation ----
  # Calculate r-squareds (training set)
  result_data <- result_data %>% 
    group_by(item_nbr) %>% 
    do(rsq_train = safely(cor)(pluck(.$sales_train, 1),
                               pluck(.$predictions_train, 1),
                               use = "pairwise.complete.obs")$result ^ 2) %>% 
    inner_join(result_data, ., by = "item_nbr")
  
  # Calculate r-squareds (test set)
  result_data <- result_data %>% 
    group_by(item_nbr) %>% 
    do(rsq_test = safely(cor)(pluck(.$sales_test, 1),
                              pluck(.$predictions_test, 1),
                              use = "pairwise.complete.obs")$result ^ 2) %>% 
    inner_join(result_data, ., by = "item_nbr")
  
  print(paste("Mean training set R^2:",
              result_data$rsq_train %>% unlist() %>% mean() %>% substr(1, 5)))
  print(paste("Mean test set R^2:",
              result_data$rsq_test %>% unlist() %>% mean() %>% substr(1, 5)))
  
  return(result_data)
  
  # Keep only the needed variables
  #result_data <- result_data %>% 
  #  select(item_nbr, rsq_train, mape_train, rsq_test, mape_test)
}

make_importance_plots <- function(model_df, data_df, lambda = FALSE){
  for(i in 1:nrow(model_df)){
    # Calculate variable importances
    variable_importance <- varImp(model_df$model[[i]],
                                    lambda = data_df$lambda,
                                  scale = TRUE)
    
    if(class(variable_importance) == "varImp.train"){
      variable_importance <- variable_importance$importance
    }
    
    variable_importance <- variable_importance %>% 
      mutate(Variable = row.names(.),
             Importance = as.numeric(Overall)) %>%
      select(-Overall) %>%
      arrange(-Importance) %>% 
      mutate(Variable = gsub("factor(month, levels = 1:12)",
                             "",
                             .$Variable,
                             fixed = TRUE) %>%
               reorder(Importance)) 
    
    importance_plot[[i]] <- variable_importance %>% 
      ggplot(aes(x = Variable,
                 y = Importance)) +
      geom_col() +
      coord_flip() +
      ggtitle(paste0("Product ", model_df$item_nbr[[i]], ", ",
                     "test set R^2 ", data_df$rsq_test[[i]] %>% substr(1, 5)))
  }
  
  return(importance_plot)
}

do.call(grid.arrange, list(grobs = make_importance_plots(linear_models, 
                                                         make_predictions(linear_models)),
                           top = "Standardized variable importances, linear model"))

do.call(grid.arrange, list(grobs = make_importance_plots(ridge_models, 
                                   make_predictions(ridge_models,
                                                    lambda = TRUE),
                                   lambda = TRUE),
                           top = "Standardized variable importances, ridge model"))

do.call(grid.arrange, list(grobs = make_importance_plots(xgb_models, 
                                   make_predictions(xgb_models)),
                           top = "Standardized variable importances, XGBoost model"))
