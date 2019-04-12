library(MASS)
library(dplyr)
library(broom)
library(purrr)
library(tibble)
library(glmnet)
library(ggplot2)
library(dummies)
library(data.table)

options(scipen = 1e6)
memory.limit(1e9)
sales <- fread("train.csv", nrows = 3939438)
holidays <- fread("holidays_events.csv")

# Form training and test sets
train <- sales %>%
  arrange(date, store_nbr, item_nbr) %>% 
  head(3000000)
test <- sales %>%
  arrange(date, store_nbr, item_nbr) %>%
  tail(939437)

stores_and_items <- train %>%
  group_by(store_nbr, item_nbr) %>%
  summarise()

# Amount of stores, products & their combinations
length(unique(stores_and_items$store_nbr))
length(unique(stores_and_items$item_nbr))
nrow(stores_and_items)

# Mutate holidays to binary for each date
holidays_bin <- as.data.frame(cbind(holidays$date,
                                    dummy(holidays$type),
                                    dummy(holidays$locale),
                                    as.numeric(holidays$transferred)))

colnames(holidays_bin) <- c("date", "additional", "bridge", "event", "holiday",
                            "transfer", "workday", "local", "national",
                            "regional", "transferred")

holidays_bin$date <- as.character(holidays_bin$date)
holidays_bin[, 2:11] <- apply(holidays_bin[, 2:11], 2, as.numeric)

# Join holidays to sales data
train <- left_join(train, holidays_bin, by = "date")
test <- left_join(test, holidays_bin, by = "date")
# Replace NAs with zeros
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# Add weekdays
train$weekday <- as.factor(weekdays(as.Date(train$date)))
test$weekday <- as.factor(weekdays(as.Date(test$date)))

# Convert df to tibble
stores_and_items <- stores_and_items %>%
  as_tibble()

# Temp dataset to run faster
tr <- train %>% 
  filter(store_nbr %in% c(1, 25), item_nbr %in% c(103665, 105574))


tr <- train %>%
  arrange(store_nbr, item_nbr) %>% 
  slice(1:70000)
#################################

# Group
train_grouped <- tr %>%  ## tr
  group_by(store_nbr, item_nbr)

# Make model matrices
mm_holder <- train_grouped %>%
  do(mm = safely(model.matrix)(unit_sales ~ onpromotion + transferred +
                                    local + regional + additional + bridge +
                                    event + holiday + transfer + workday +
                                    weekday, data = .)$result)

# Add unit sales to the tibble
sales_holder <- train_grouped %>% 
  summarise(unit_sales = list(unit_sales)) %>% 
  inner_join(., mm_holder)

# Do cross validations for every store-product combination
temp2 <- sales_holder %>% 
  group_by(store_nbr, item_nbr) %>% 
  # pluck() to get the first element inside each list
  do(cv = safely(cv.glmnet)(pluck(.$mm, 1), 
                            pluck(.$unit_sales, 1), alpha = 1, nlambda = 100)$result) %>% 
  inner_join(sales_holder, .)

# Extract lambdas to a separate column
temp <- temp2 %>%
  # Select lamdas from lists and convert NULLs to NAs
  add_column(as.numeric(as.character(lapply(temp2$cv, `[[`, 9))))
colnames(temp)[6] <- "lambda"

# Do regressions
regression <- sales_holder %>% 
  group_by(store_nbr, item_nbr) %>% 
  do(lasso = safely(glmnet)(pluck(.$mm, 1),
                            pluck(.$unit_sales, 1),
                            type.gaussian = "naive",
                            lambda = .$lambda)$result) %>% 
  inner_join(temp, .)

# Predict by using training data to obtain accuracy
predictions <- regression %>% 
  group_by(store_nbr, item_nbr) %>% 
  do(predictions = safely(predict)(pluck(.$lasso, 1),
                                   s = .$lambda,
                                   newx = pluck(.$mm, 1))$result) %>% 
  inner_join(regression, .)

# Find the lengths of the actual values where predictions are missing
len <- sapply(predictions$unit_sales[which(predictions$predictions == "NULL")], length)
# Make NAs that have same lengths as the unit_sales
nas <- sapply(len, function(x) rep(NA, x))
# Replace single NAs with multiple NAs that are same length as the corresponding unit_sales
predictions$predictions[which(predictions$predictions == "NULL")] <- nas

# Function for finding the p-values of the correlations
find_pvalue <- function(x, y){
  ifelse(all(is.na(y[[1]])), NA, 
  safely(cor.test)(x,y)$result$p.value
  )
}

# Calculate correlations and p-values for training set
predictions <- predictions %>%
  group_by(store_nbr, item_nbr) %>% 
  summarise(r = safely(cor)(pluck(unit_sales, 1),
                            pluck(predictions, 1),
                            use = "pairwise.complete.obs")$result,
            p = find_pvalue(pluck(unit_sales, 1),
                         pluck(predictions, 1))) %>% 
  inner_join(predictions, .)

# Add to stores_and_items tibble
stores_and_items <- predictions %>% 
  select(store_nbr, item_nbr, r, p) %>% 
  rename(r_squared_train = r,
         p_value_train = p) %>% 
  right_join(stores_and_items, by = c("store_nbr", "item_nbr"))

#################################

te <- test %>% 
  filter(store_nbr %in% c(1, 25), item_nbr %in% c(103665, 105574))

# Group
test_grouped <- te %>%  ## tr
  group_by(store_nbr, item_nbr)

# Make model matrices for test set
mm_holder_test <- test_grouped %>%
  do(mm = safely(model.matrix)(unit_sales ~ onpromotion + transferred +
                                 local + regional + additional + bridge +
                                 event + holiday + transfer + workday +
                                 weekday, data = .)$result)

# Add unit sales to tibble
sales_holder_test <- test_grouped %>% 
  summarise(unit_sales = list(unit_sales)) %>% 
  inner_join(., mm_holder_test)

# Attach lasso models & lambdas to test set
sales_holder_test <- predictions %>% 
  select(store_nbr, item_nbr, lambda, lasso) %>% 
  inner_join(sales_holder_test)

# Calculate predictions for test set
sales_holder_test <- sales_holder_test %>% 
  group_by(store_nbr, item_nbr) %>% 
  do(predictions = safely(predict)(pluck(.$lasso, 1),
                                   s = .$lambda,
                                   newx = pluck(.$mm, 1))$result) %>% 
  inner_join(sales_holder_test, .)

# Find the lengths of the actual values where predictions are missing
len <- sapply(sales_holder_test$unit_sales[which(sales_holder_test$predictions == "NULL")], length)
# Make NAs that have same lengths as the unit_sales
nas <- sapply(len, function(x) rep(NA, x))
# Replace single NAs with multiple NAs that are same length as the corresponding unit_sales
sales_holder_test$predictions[which(sales_holder_test$predictions == "NULL")] <- nas

# Calculate correlations and p-values for test set
sales_holder_test <- sales_holder_test %>%
  group_by(store_nbr, item_nbr) %>% 
  summarise(r = safely(cor)(pluck(unit_sales, 1),
                            pluck(predictions, 1),
                            use = "pairwise.complete.obs")$result,
            p = find_pvalue(pluck(unit_sales, 1),
                            pluck(predictions, 1))) %>% 
  inner_join(sales_holder_test, .)

# Add to stores_and_items tibble
stores_and_items <- sales_holder_test %>% 
  select(store_nbr, item_nbr, r, p) %>% 
  rename(r_squared_test = r,
         p_value_test = p) %>% 
  right_join(stores_and_items, by = c("store_nbr", "item_nbr"))

##########################################################################
# Plotting
par(mfrow = c(2, 1))
hist(stores_and_items$r_squared_train, breaks = 100, main = "R-squared of training set")
hist(stores_and_items$r_squared_test, breaks = 100, main = "R-squared of test set")

hist(stores_and_items$p_value_train, breaks = 100, main = "P-values of training set")
hist(stores_and_items$p_value_test, breaks = 100, main = "P-values of test set")

# Function for plotting predictions and actuals
plot_predictions <- function(i = NA, store = NA, item = NA){
  if(!is.na(i)){temp_predictions <- test %>%
    filter(
      store_nbr == stores_and_items[i, 1]$store_nbr,
      item_nbr == stores_and_items[i, 2]$item_nbr)
  
  temp_predictions$prediction <- predict(stores_and_items$formula[[i]],
                                         newdata = temp_predictions)
  ggplot(temp_predictions, aes(as.Date(date))) +
    geom_line(aes(y = unit_sales, group = 1), size = 1.5) +
    geom_line(aes(y = prediction, group = 2), size = 1.5, col = "#01BFC4") +
    ggtitle("Predicted vs actual") +
    labs(subtitle = paste0("Store id: ", stores_and_items$store_nbr[i],
                           ", Prodct id: ", stores_and_items$item_nbr[i],
                           ", R-squared ", round(stores_and_items$r_squared_test[i], 2),
                           ", P-value ", round(stores_and_items$pvalue_test[i], 3))) +
    xlab("Date") +
    ylab("Sales")
  
  } else {
    temp_predictions <- test %>%
      filter(
        store_nbr == store,
        item_nbr == item)
    which_row <- which(stores_and_items$store_nbr == store & 
                         stores_and_items$item_nbr == item)
    
    temp_predictions$prediction <- predict(stores_and_items[[which_row, "formula"]])
    
    ggplot(temp_predictions, aes(as.Date(date))) +
      geom_line(aes(y = unit_sales, group = 1), size = 1.5) +
      geom_line(aes(y = prediction, group = 2), size = 1.5, col = "#01BFC4") +
      ggtitle("Predicted vs actual") +
      labs(subtitle = paste0("Store id: ", store,
                             ", Prodct id: ", item,
                             ", R-squared ", round(stores_and_items[
                               which_row,"r_squared_test"]$r_squared_test, 2),
                             ", P-value ", round(stores_and_items[
                               which_row, "pvalue_test"]$pvalue_test, 3))) +
      xlab("Date") +
      ylab("Sales")
  }
}
