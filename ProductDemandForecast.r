library(data.table)
library(dplyr)
library(lubridate)
library(MASS)
library(broom)

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
holidays_bin <- holidays %>%
  mutate(
    additional = ifelse(type == "Additional", 1, 0),
    bridge = ifelse(type == "Bridge", 1, 0),
    event = ifelse(type == "Event", 1, 0),
    holiday = ifelse(type == "Holiday", 1, 0),
    transfer = ifelse(type == "Transfer", 1, 0),
    workday = ifelse(type == "Work Day", 1, 0),

    local = ifelse(locale == "Local", 1, 0),
    regional = ifelse(locale == "Regional", 1, 0),
    transferred = as.numeric(transferred)
  ) %>%
  select(-type, -locale, -locale_name, -description)

# Join holidays to sales data
train <- left_join(train, holidays_bin, by = "date")
test <- left_join(test, holidays_bin, by = "date")
# Check if any other fields are NA & replace NAs with zeros
apply(train, 2, function(x) any(is.na(x)))
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# Add weekdays
train$weekday <- as.factor(weekdays(as.Date(train$date)))
test$weekday <- as.factor(weekdays(as.Date(test$date)))

# Add fields for accuracy measures & format into tibble to store formulas
stores_and_items <- stores_and_items %>%
  mutate(
    r_squared_train = NA,
    r_squared_test = NA,
    pvalue_train = NA,
    pvalue_test = NA,
    formula = NA
  ) %>%
  as_tibble()

# Make regressions using training data
time <- Sys.time()
for (i in 1:nrow(stores_and_items)) {
  # Select one item from one store at a time
  temp <- train %>%
    filter(
      store_nbr == stores_and_items[i, 1]$store_nbr,
      item_nbr == stores_and_items[i, 2]$item_nbr
    )

  # If not enough data, go to next
  if (nrow(temp) == 0 || nrow(temp) == 1) {next}

  # Do linear models for forecasting demand
  lm <- tryCatch(lm(unit_sales ~
  onpromotion + transferred + local + regional + additional +
    bridge + event + holiday + transfer + workday + weekday,
  data = temp),
  error = function(x) {lm <- NA})

  # Skip stepwise selection and set forecast as mean sales if AIC is -infinite
  if (is.na(lm) || AIC(lm) == -Inf) {
    formula <- sum(temp$unit_sales) /
      as.numeric(max(as.Date(temp$date)) - min(as.Date(temp$date)))
    stores_and_items[i, "formula"] <- formula
    next
  }
  lm_stepwise <- stepAIC(lm, trace = F)

  # Select the model with better r-squared
  ifelse(glance(lm_stepwise)$r.squared > glance(lm_stepwise)$r.squared,
    formula <- temp %>%
      do(model = lm),
    formula <- temp %>%
      do(model = stepAIC(lm, trace = F))
  )

  stores_and_items[i, "r_squared_train"] <- summary(lm)$r.squared
  stores_and_items[i, "pvalue_train"] <- glance(lm)$p.value
  stores_and_items[i, "formula"] <- formula
  i_holder[i] <- i
}
Sys.time() - time

time <- Sys.time()
# Test the regression on test data
for (i in 1:nrow(stores_and_items)) {
  temp_predict <- test %>%
    filter(
      store_nbr == stores_and_items[i, 1]$store_nbr,
      item_nbr == stores_and_items[i, 2]$item_nbr
    )

  # If not enough data to predict or formula was made on lines 78-81, go to next
  if (nrow(temp_predict) == 0 || nrow(temp_predict) == 1) {next}
  if (is.numeric(stores_and_items$formula[[i]])) {next}

  # Get the formula from a cell and use it to predict
  temp_predict$prediction <- predict(stores_and_items$formula[[i]],
    newdata = temp_predict
  )
  if (is.na(temp_predict$prediction)) {next}

  # Calculate the r^2 and p-value for the test set
  stores_and_items[i, "r_squared_test"] <- tryCatch(unname(cor.test(
    temp_predict$unit_sales, temp_predict$prediction)$estimate)^2,
  error = function(x) {stores_and_items[i, "r_squared_test"] <- NA})

  stores_and_items[i, "pvalue_test"] <- tryCatch(unname(cor.test(
    temp_predict$unit_sales, temp_predict$prediction)$p.value),
  error = function(x) {stores_and_items[i, "r_squared_test"] <- NA})
}
Sys.time() - time
