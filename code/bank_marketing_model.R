# =========================================
# Bank Marketing Machine Learning Project
# Author: Nguyen Hoang Van Nhi
# Description: Predict customer response using ML models
# =========================================
# 0. Libraries
  
library(readxl)
library(dplyr)
library(e1071)
library(caret)
library(car)
library(pROC)
library(rpart)
library(rpart.plot)
library(themis)
library(randomForest)

set.seed(123)

# 1. Data Loading
  
 bank_data <- read_excel("bank_marketing_dataset.xlsx")

# 2. Data Cleaning & Preparation
  
  bank_prep <- bank_data

## 2.1 Handle "unknown" categories

unknown_vars <- c("default", "housing", "loan", "education")

for (v in unknown_vars) {
  bank_prep[[v]][bank_prep[[v]] == "unknown"] <- NA
}

### Drop default (too many missing, low business value)

bank_prep$default <- NULL

## 2.2 Handle pdays (999 = not previously contacted)

bank_prep$pdays[bank_prep$pdays == 999] <- NA

### Create previous contact indicator

bank_prep$previous_contact <- ifelse(bank_prep$previous > 0, 1, 0)
bank_prep$previous_contact <- factor(bank_prep$previous_contact)


### Impute pdays based on customers with previous contact only
median_pdays <- median(
  bank_prep$pdays[bank_prep$previous > 0],
  na.rm = TRUE
)

bank_prep$pdays[
  bank_prep$previous > 0 & is.na(bank_prep$pdays)
] <- median_pdays


## 2.3 Remove data leakage variable

bank_prep$duration <- NULL

## 2.4 Drop remaining missing values in key categorical variables

bank_prep <- bank_prep %>%
  filter(
    !is.na(housing),
    !is.na(loan),
    !is.na(education)
  )
### Remaining missing categorical values are removed due to low proportion and lack of reliable imputation strategy


### Ensure response is factor before modelling and SMOTE
bank_prep$response <- factor(
  bank_prep$response,
  levels = c("no", "yes")
)


# 3. Feature Engineering & Transformation

## Log-transform skewed behavioural variables

bank_prep$log_campaign <- log(bank_prep$campaign + 1)
bank_prep$log_previous <- log(bank_prep$previous + 1)


## Standardize macroeconomic variables
### NOTE: Scaling here is exploratory for distribution inspection. Final scaling is applied within the training set only and propagated to the test set during preprocessing (recipe step) to avoid data leakage.

scale_vars <- c(
  "emp_var_rate", "cons_price_idx",
  "cons_conf_idx", "euribor3m", "nr_employed"
)

bank_prep[scale_vars] <- scale(bank_prep[scale_vars])


## Standardize age for model stability
### Final standardization is performed post train-test split

bank_prep$age_z <- scale(bank_prep$age)



# 4. Logistic Regression – VIF Diagnostics

library(car)

## Ensure categorical variables are treated as factors

factor_vars <- c(
  "job", "marital", "education",
  "housing", "loan",
  "contact", "month", "poutcome",
  "response"
)

bank_prep[factor_vars] <- lapply(bank_prep[factor_vars], factor)

## Initial logistic regression model for multicollinearity diagnostics

logit_vif_model <- glm(
  response ~ age_z + log_campaign + log_previous + pdays +
    emp_var_rate + cons_price_idx + cons_conf_idx +
    euribor3m + nr_employed +
    job + marital + education + housing + loan +
    contact + month + poutcome,
  data = bank_prep,
  family = binomial
)

## Compute Generalised Variance Inflation Factors (GVIF)

vif_raw <- vif(logit_vif_model)

## Create interpretable VIF table (adjusted GVIF)

vif_table <- data.frame(
  Variable = rownames(vif_raw),
  GVIF = round(vif_raw[, "GVIF"], 2),
  Df   = vif_raw[, "Df"],
  GVIF_adj = round(vif_raw[, "GVIF^(1/(2*Df))"], 2)
)

vif_table

### High multicollinearity among macroeconomic variables is expected due to strong economic co-movement (e.g., euribor3m, nr_employed, emp_var_rate).Reduced model specifications are subsequently tested to prioritise ranking performance, numerical stability, and interpretability rather than statistical significance alone.

## Final variable set after VIF diagnostics and business consideration

final_logit_vars <- c(
  "age_z",
  "log_campaign",
  "log_previous",
  "pdays",
  "previous_contact",
  "cons_price_idx",
  "cons_conf_idx",
  "euribor3m",
  "job",
  "marital",
  "education",
  "housing",
  "loan",
  "contact",
  "month",
  "poutcome",
  "response"
)

# 5. Train–Test Split (Stratified)
 
## Create modelling dataset (final variables only)

model_data <- bank_prep[, final_logit_vars]

model_data$response <- factor(
  model_data$response,
  levels = c("no", "yes")
)

  train_index <- createDataPartition(
    model_data$response,
    p = 0.7,
    list = FALSE
  )

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# 6. Handle Class Imbalance – SMOTE (TRAINING ONLY)

library(recipes)
library(themis)

## Base preprocessing (NO SMOTE)
base_recipe <- recipe(response ~ ., data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

## SMOTE recipe (training only)
smote_recipe <- base_recipe %>%
  step_smote(response)

## Prep recipes
prep_base  <- prep(base_recipe)
prep_smote <- prep(smote_recipe)

## Apply preprocessing
train_data_smote <- bake(prep_smote, new_data = NULL)
test_data_processed <- bake(prep_base, new_data = test_data)

### SMOTE is applied ONLY to training data. Test set remains untouched to reflect real-world class distribution

## Check balance
prop.table(table(train_data_smote$response))



# 7. Logistic Regression – Final Model
## 7.1 Model Estimation (Training Data Only)
### Fit logistic regression on SMOTE-balanced training data
logit_model <- glm(
  response ~ .,
  data = train_data_smote,
  family = binomial
)

summary(logit_model)

## 7.2 Prediction on Test Data (Using the Same Recipe)

### Predict probabilities on test set
test_prob <- predict(
  logit_model,
  newdata = test_data_processed,
  type = "response"
)
## 7.3 Classification with Business-Driven Threshold
### Apply business-driven threshold

test_pred_035 <- ifelse(test_prob > 0.35, "yes", "no")
test_pred_035 <- factor(test_pred_035, levels = c("no", "yes"))

### Threshold set to 0.35 to prioritize recall over precision, reflecting higher business cost of missing potential subscribers


## 7.4 Confusion Matrix & Performance Metrics
library(caret)

cm_logit <- confusionMatrix(
  test_pred_035,
  test_data$response,
  positive = "yes"
)

cm_logit

## 7.5 ROC Curve & AUC
library(pROC)

roc_logit <- roc(
  test_data$response,
  test_prob,
  levels = c("no", "yes"),
  direction = "<"
)

auc_logit <- auc(roc_logit)

auc_logit

### Plot ROC curve
plot(
  roc_logit,
  main = "ROC Curve – Logistic Regression"
)
## 7.6 Calibration – Brier Score

brier_logit <- mean(
  (as.numeric(test_data$response == "yes") - test_prob)^2
)

brier_logit


# 8. Decision Tree – CV-based Pruning
  
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)

## 8.1 Prepare data

tree_data <- bank_prep
tree_data$response <- factor(tree_data$response, levels = c("no", "yes"))

set.seed(123)
tree_index <- createDataPartition(
  tree_data$response,
  p = 0.7,
  list = FALSE
)

tree_train <- tree_data[tree_index, ]
tree_test  <- tree_data[-tree_index, ]

## 8.2 Define model formula

tree_formula <- response ~ .

## 8.3 Train full decision tree

tree_full <- rpart(
  tree_formula,
  data = tree_train,
  method = "class",
  control = rpart.control(
    cp = 0.001,     # grow a large tree first
    minsplit = 20,
    maxdepth = 5
  )
)

## 8.4 Cross-validation results

printcp(tree_full)
plotcp(tree_full)

## 8.5 Prune tree using minimum CV error

best_cp <- tree_full$cptable[
  which.min(tree_full$cptable[, "xerror"]), "CP"
]

best_cp

tree_pruned <- prune(tree_full, cp = best_cp)

## 8.6 Plot final pruned tree

rpart.plot(
  tree_pruned,
  type = 2,
  extra = 106,
  under = TRUE,
  fallen.leaves = TRUE,
  box.palette = "GnBu",
  main = "Decision Tree (minsplit = 20, maxdepth = 5)"
)

## 8.7 Variable importance

tree_pruned$variable.importance

## 8.8 Prediction on test set

tree_prob <- predict(
  tree_pruned,
  newdata = tree_test,
  type = "prob"
)[, "yes"]

tree_pred <- predict(
  tree_pruned,
  newdata = tree_test,
  type = "class"
)

### Calibration – Brier Score (Decision Tree)
brier_tree <- mean(
  (as.numeric(tree_test$response == "yes") - tree_prob)^2
)

brier_tree

## 8.9 Confusion matrix

cm_tree <- confusionMatrix(
  tree_pred,
  tree_test$response,
  positive = "yes"
)

cm_tree

## 8.10 ROC curve and AUC

roc_tree <- roc(
  tree_test$response,
  tree_prob,
  levels = c("no", "yes"),
  direction = "<"
)

auc_tree <- auc(roc_tree)
auc_tree

plot(
  roc_tree,
  main = "ROC Curve – Decision Tree"
)


# 9. Random Forest – Cost-Sensitive Benchmark
  
### Remove variables with structural NA (pdays contains NA by design)
rf_vars <- setdiff(
  names(tree_train),
  c("pdays")
)
### Subset training and test data
rf_train <- tree_train[, rf_vars]
rf_test  <- tree_test[, rf_vars]

### Remove rows with NA (Random Forest requirement)
rf_train <- na.omit(rf_train)
rf_test  <- na.omit(rf_test)

### Sanity check
colSums(is.na(rf_train))
colSums(is.na(rf_test))

## Random Forest Model Estimation (Cost-Sensitive)

class_counts <- table(rf_train$response)
min_class <- min(class_counts)

set.seed(123)

rf_model <- randomForest(
  response ~ .,
  data = rf_train,
  ntree = 300,
  mtry = floor(sqrt(ncol(rf_train) - 1)),
  sampsize = c("no" = min_class, "yes" = min_class),
  importance = TRUE
)

### Predict probabilities
rf_prob <- predict(
  rf_model,
  rf_test,
  type = "prob"
)[, "yes"]

### Class prediction (default threshold = 0.5)
rf_pred <- ifelse(rf_prob > 0.5, "yes", "no")
rf_pred <- factor(rf_pred, levels = c("no", "yes"))

## Model Evaluation – Confusion Matrix
library(caret)

cm_rf <- confusionMatrix(
  rf_pred,
  rf_test$response,
  positive = "yes"
)

cm_rf

## ROC Curve and AUC
library(pROC)

roc_rf <- roc(
  rf_test$response,
  rf_prob,
  levels = c("no", "yes"),
  direction = "<"
)

auc_rf <- auc(roc_rf)
auc_rf

### Random Forest
brier_rf <- mean(
  (as.numeric(rf_test$response == "yes") - rf_prob)^2
)
brier_rf

### Plot ROC
plot(
  roc_rf,
  main = "ROC Curve – Random Forest"
)
## Lift Analysis (Top-Decile Performance)
library(dplyr)

lift_rf <- data.frame(
  actual = rf_test$response,
  prob = rf_prob
)

### Highest probability = decile 1
lift_rf$decile <- ntile(desc(lift_rf$prob), 10)

baseline_rf <- mean(lift_rf$actual == "yes")

lift_rf_summary <- lift_rf %>%
  group_by(decile) %>%
  summarise(
    positives = sum(actual == "yes"),
    total = n(),
    capture_rate = positives / sum(lift_rf$actual == "yes"),
    precision = positives / total,
    lift = precision / baseline_rf
  )

### Top 10% and 20%
lift_rf_summary %>% filter(decile %in% c(1, 2))

### Variable Importance 
varImpPlot(
  rf_model,
  main = "Variable Importance – Random Forest"
)
# 10. Lift Analysis (Logistic – Business Focus)
### Set seed for reproducibility of lift and final comparison
set.seed(123)

## 10.1 Construct Lift Dataset
### Lift Analysis – Logistic Regression

lift_logit <- data.frame(
  actual = test_data$response,
  prob   = test_prob
)

### Highest predicted probability = Decile 1
lift_logit$decile <- ntile(desc(lift_logit$prob), 10)

### Baseline response rate (random targeting)
baseline_rate <- mean(lift_logit$actual == "yes")

## 10.2 Compute Lift Metrics
lift_logit_summary <- lift_logit %>%
  group_by(decile) %>%
  summarise(
    positives    = sum(actual == "yes"),
    total        = n(),
    capture_rate = positives / sum(lift_logit$actual == "yes"),
    precision    = positives / total,
    lift         = precision / baseline_rate
  )

lift_logit_summary

## 10.3 Business-Focused Lift Summary (Top Segments)

lift_business <- lift_logit_summary %>%
  filter(decile %in% c(1, 2)) %>%
  select(decile, capture_rate, lift)

lift_business

## Lift Chart

  plot(
    lift_logit_summary$decile,
    lift_logit_summary$lift,
    type = "b",
    xlab = "Decile (1 = Highest Probability)",
    ylab = "Lift",
    main = "Lift Chart – Logistic Regression"
  )

### Final Model Comparison – Performance & Business KPIs  
  final_compare_full <- data.frame(
    Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
    AUC   = c(auc_logit, auc_tree, auc_rf),
    Brier = c(brier_logit, brier_tree, brier_rf),
    Lift_Top20 = c(
      lift_business %>% filter(decile == 2) %>% pull(lift),
      NA,
      lift_rf_summary %>% filter(decile == 2) %>% pull(lift)
    )
  )
  
  final_compare_full
  

# END OF SCRIPT
