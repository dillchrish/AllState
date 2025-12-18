
library(vroom)
library(tidymodels)
tidymodels_prefer()
library(dplyr)
library(readr)
library(xgboost)



train <- vroom("/Users/dylan/Downloads/train.csv")
test <- vroom("/Users/dylan/Downloads/test.csv")
sample_submission <- vroom("/Users/dylan/Downloads/sample_submission.csv")

allstate_rec <- recipe(loss ~ ., data = train) %>%
  update_role(id, new_role = "ID") %>%
  step_zv(all_predictors())

rf_spec <- rand_forest(
  mtry  = 20,     
  min_n = 5,
  trees = 500
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(allstate_rec)

rf_fit     <- rf_wf %>% fit(data = train)

allstate_rec <- recipe(loss ~ ., data = train) %>%
  update_role(id, new_role = "ID") %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.005) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


glmnet_spec <- linear_reg(
  penalty = 0.01,  
  mixture = 0.5    
) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

glmnet_wf <- workflow() %>%
  add_model(glmnet_spec) %>%
  add_recipe(xgb_glmnet_rec)

glmnet_fit <- glmnet_wf %>% fit(data = train)

xgb_spec <- boost_tree(
  trees          = 500,
  tree_depth     = 6,
  learn_rate     = 0.05,
  min_n          = 5,
  loss_reduction = 0,
  sample_size    = 1,
  mtry           = 20
) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(xgb_glmnet_rec)

xgb_fit    <- xgb_wf %>% fit(data = train)

rf_pred <- predict(rf_fit,     new_data = test) %>% rename(rf = .pred)
glm_pred <- predict(glmnet_fit, new_data = test) %>% rename(glmnet = .pred)
xgb_pred <- predict(xgb_fit,    new_data = test) %>% rename(xgb = .pred)

preds <- bind_cols(
  test %>% select(id),
  rf_pred,
  glm_pred,
  xgb_pred
) %>%
  mutate(loss_ensemble = (rf + glmnet + xgb) / 3)

preds_rf <- bind_cols(
  test %>% select(id),
  rf_pred
) %>%
  mutate(loss_ensemble = (rf))

preds_glmnet <- bind_cols(
  test %>% select(id),
  rf_pred,
  glm_pred,
  xgb_pred
) %>%
  mutate(loss_ensemble = (glmnet))

preds_xbg <- bind_cols(
  test %>% select(id),
  rf_pred,
  glm_pred,
  xgb_pred
) %>%
  mutate(loss_ensemble = (xgb))

sample_submission <- sample_submission %>%
  dplyr::select(id) %>%
  dplyr::left_join(
    preds_rf %>% dplyr::select(id, loss = loss_ensemble),
    by = "id"
  )

vroom_write(sample_submission, "~/Downloads/my_submission_rf.csv",
            delim = ","  )




sample_submission <- sample_submission %>%
  dplyr::select(id) %>%
  dplyr::left_join(
    preds_glmnet %>% dplyr::select(id, loss = loss_ensemble),
    by = "id"
  )

vroom_write(sample_submission, "~/Downloads/my_submission_glmnet.csv",
            delim = ","  )

sample_submission <- sample_submission %>%
  dplyr::select(id) %>%
  dplyr::left_join(
    preds_xbg %>% dplyr::select(id, loss = loss_ensemble),
    by = "id"
  )

vroom_write(sample_submission, "~/Downloads/my_submission_xgb.csv",
            delim = ","  )
