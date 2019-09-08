library(data.table) # fread function
library(zoo) # approximating NAs
library(tidyr) # unite function, replacing NAs
library(dtplyr) # converting dplyr to data.table
library(dplyr) # data wrangling
library(lubridate) # handling dates
library(purrr) # handling nested data
library(glmnet) # ridge regression
library(caret) # XGBoost model, varImp function
library(ggplot2) # plotting
library(gridExtra) # plotting multiple plots together
library(tibble) # add_column function
library(multidplyr) # parallel dplyr
library(parallel) # find out amount of cores

# Reading & transforming the data ----
# Select specific items
items_to_be_plotted <- c(1047679,
                         819932,
                         364606)

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
sales_data <- fread("train.csv")

# Select interval and aggregate to monthly
sales_data <- sales_data %>% 
  lazy_dt() %>% 
  # Selected items only
  filter(item_nbr %in% items_to_be_plotted) %>% 
  filter(date >= as.Date(last(date)) - years(4)) %>% 
  mutate(date = as.Date(date),
         year = year(date),
         month = month(date) %>% as.character(),
         # Replace missing promotions with zero
         promo = replace_na(onpromotion, 0),
         store_nbr = as.character(store_nbr)) %>% 
  group_by(year, month, item_nbr, store_nbr) %>% 
  summarise(sales = sum(unit_sales),
            promo = mean(promo)) %>% 
  # Make a date column with the first day of the months
  mutate(year_month = as.Date(paste0(year, "-", month, "-01"))) %>% 
  as_tibble()

# Use ~67/33 training/test split
split_date <- last(sales_data$year_month) - years(1)

# Combine the sales data with the oil price data
full_data <- sales_data %>% 
  left_join(oil_df)

# Make lagged sales variables
full_data <- full_data %>% 
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(store_nbr, item_nbr) %>% 
  mutate(sales_lag12 = lag(sales, 12),
         sales_lag1 = lag(sales, 1)) %>% 
  na.omit()

# Splitting into training and test sets ----
# Make training set with dates and actual sales
to_model <- full_data %>% 
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(item_nbr) %>% 
  filter(year_month <= split_date) %>% 
  summarise(year_month_train = list(year_month),
            sales_train = list(sales))

# Make test set with dates and actual sales
to_model <- full_data %>% 
  arrange(item_nbr, store_nbr, year, as.numeric(month)) %>% 
  group_by(item_nbr) %>% 
  filter(year_month > split_date) %>% 
  summarise(year_month_test = list(year_month),
            sales_test = list(sales)) %>% 
  inner_join(to_model, .)

# Split into training and test sets by date
train <- full_data %>% 
  filter(year_month <= split_date)
  
test <- full_data %>% 
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

# Do model matrices for testing data (dummy variables etc.)
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
# Do cross validations to extract lamdas for the ridge models
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
# Create cluster for parallel processing
cluster <- new_cluster(detectCores())
cluster %>%
  cluster_library("purrr") %>%
  cluster_library("caret")

time <- Sys.time() # ~15 min
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
                             # smaller value prevents overfitting, 0-inf
                             max_depth = c(6, 10, 15, 25),
                             # smaller value prevents overfitting, 0-inf
                             eta = c(0.01, 0.05, 0.1, 0.2, 0.5),
                             # higher value = more conservative, 0-inf
                             gamma = c(0, 5),
                             # 0-1
                             colsample_bytree = c(0.1, 0.3, 0.5, 0.8, 1),
                             # higher value = more conservative, 0-inf
                             min_child_weight = 1,
                             # smaller value prevents overfitting, 0-1,
                             subsample = c(0.5, 1)),
                           allowParallel = TRUE)$result) %>% 
  collect()
(time <- Sys.time() - time)

# Model evaluation ----
# Function to make predictions and calculate accuracy measures for both sets
make_predictions <- function(model_df, lambda = FALSE){
  # Make predictions using the models (training set)
  # Handle ridge models separately as they have different arguments
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
  # Handle linear and XGBoost models
  } else {
    # Make predictions using the models (training set)
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
  
  # Print mean R-squareds
  print(paste("Mean training set R^2:",
              result_data$rsq_train %>% unlist() %>% mean() %>% substr(1, 5)))
  print(paste("Mean test set R^2:",
              result_data$rsq_test %>% unlist() %>% mean() %>% substr(1, 5)))
  
  return(result_data)
}

# Make predictions for each model
pred_Linear <- make_predictions(linear_models)
pred_Ridge <- make_predictions(ridge_models, lambda = TRUE)
pred_XGB <- make_predictions(xgb_models)

# Function for plotting variable importances
importance_plot <- list()
make_importance_plots <- function(model_df, data_df, lambda = FALSE){
  # Loop through all models
  for(i in 1:nrow(model_df)){
    # Calculate variable importances
    variable_importance <- varImp(model_df$model[[i]],
                                    lambda = data_df$lambda,
                                  scale = TRUE)
    # XGBoost variable importances are handled differently
    if(class(variable_importance) == "varImp.train"){
      variable_importance <- variable_importance$importance
    }
    
    # Convert type while keeping names and arrange
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
    
    # Produce plots into a list
    importance_plot[[i]] <- variable_importance %>% 
      ggplot(aes(x = Variable,
                 y = Importance)) +
      geom_col() +
      coord_flip() +
      ggtitle(paste0("Product ", model_df$item_nbr[[i]], ", ",
                     "test R^2 ", data_df$rsq_test[[i]] %>% substr(1, 5))) +
      theme_light()
  }
  
  return(importance_plot)
}

# Collect variable importance plots to a list
plots <- c(make_importance_plots(linear_models, pred_Linear),
           make_importance_plots(ridge_models, pred_Ridge, lambda = TRUE),
           make_importance_plots(xgb_models, pred_XGB))

# Plot variable importances together
do.call(grid.arrange, list(
  arrangeGrob(grobs = plots[1:3], top = "Linear models"),
  arrangeGrob(grobs = plots[4:6], top = "Ridge models"), 
  arrangeGrob(grobs = plots[7:9], top = "XGBoost models"), 
  ncol = 3,
  top = "Standardized variable importances"))

# Unnest and aggregate actuals and predicitons to plottable format
unnest_predictions <- function(data_df){
  # Training set
  unnested <- data_df %>% 
    # Carefully remove lambda column if it exists
    select(-matches("lambda")) %>% 
    unnest(Date = year_month_train,
           Actual = sales_train,
           Prediction = predictions_train) %>% 
    rbind(data_df %>% 
            select(-matches("lambda")) %>%
            unnest(Date = year_month_train,
                   Actual = sales_train,
                   Prediction = predictions_train)) %>% 
    group_by(item_nbr, Date) %>% 
    summarise(Actual = sum(Actual),
              Prediction = sum(Prediction)) %>% 
    # Set negative predictions to zero
    mutate(Prediction = ifelse(Prediction < 0, 0, Prediction))
  
  # Test set
  unnested <- unnested %>% 
    rbind(data_df %>% 
            select(-matches("lambda")) %>%
            unnest(Date = year_month_test,
                   Actual = sales_test,
                   Prediction = predictions_test) %>% 
            rbind(data_df %>% 
                    select(-matches("lambda")) %>%
                    unnest(Date = year_month_test,
                           Actual = sales_test,
                           Prediction = predictions_test)) %>% 
            group_by(item_nbr, Date) %>% 
            summarise(Actual = sum(Actual),
                      Prediction = sum(Prediction))) %>% 
    # Set negative predictions to zero
    mutate(Prediction = ifelse(Prediction < 0, 0, Prediction)) %>% 
    # Filter last month since full data of that month is not available
    filter(Date != last(Date))
  
  return(unnested)
}

# Get the model names from the current environment
model_names <- ls()[startsWith(ls(), "pred_")]

# Loop for plotting actuals vs predictions
prediction_plot <- list()
for(i in 1:length(model_names)){
  data_df <- model_names[i]
  prediction_plot[[i]] <- unnest_predictions(get(data_df)) %>% 
    ggplot(aes(x = Date)) +
    geom_line(aes(y = Actual), size = 1) +
    geom_line(aes(y = Prediction), color = "#00BFC4", size = 1) +
    geom_vline(xintercept = split_date, color = "red", alpha = 0.5, size = 1) +
    # Disable scientific notation for sales
    scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
    # Plot each product horizonttally
    facet_grid(rows = vars(item_nbr)) +
    # Get titles from current environment variable names
    ggtitle(strsplit(ls()[startsWith(ls(), "pred_")][i], "_")[[1]][2]) +
    ylab("Sales (pcs)") +
    theme_light()
  }

# Plot actuals vs predictions for each model and product together
do.call(grid.arrange, list(grobs = prediction_plot,
                           ncol = 3,
                           top = paste0("Predictions (blue) vs actuals ",
                                        "for different models and products, ",
                           "red line separates training and test sets")))