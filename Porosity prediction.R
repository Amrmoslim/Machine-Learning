setwd("C:/R/Porosity prediction")

library(readxl)
library(tidyverse)   # Loads dplyr, ggplot2, purrr, and other useful packages
library(tidymodels)  # Loads parsnip, rsample, recipes, yardstick
library(skimr)       # Quickly get a sense of data
library(knitr)       # Pretty HTML Tables


Data_Clust <- read_excel("Data_Clust.xlsx")

Data_Clust %>% head() %>%  kable()

Data_Clust%>% skim()

## Train/Test Split

set.seed(seed = 1972) 

train_test_split <-
    rsample::initial_split(
        data = Data_Clust,     
        prop = 0.80   
    ) 

train_test_split


train_tbl <- train_test_split %>% training() 
test_tbl  <- train_test_split %>% testing() 

recipe_simple <- function(dataset) {
    recipe(Porosity ~ ., data = dataset) %>%
        prep(data = dataset)
}
# Prepped the train_tbl
recipe_prepped <- recipe_simple(dataset = train_tbl)

train_baked <- bake(recipe_prepped, new_data = train_tbl)
test_baked  <- bake(recipe_prepped, new_data = test_tbl)

#Machine Learning and Performance
#
logistic_glm <- logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Porosity ~ ., data = train_baked)

# Assess Performance
predictions_glm <- logistic_glm %>%
    predict(new_data = test_baked) %>%
    bind_cols(test_baked %>% select(Porosity))

predictions_glm %>% head() %>% kable()

predictions_glm %>%
    conf_mat(Porosity, .pred_class) %>%
    pluck(1) %>%
    as_tibble() %>%
    
    # Visualize with ggplot
    ggplot(aes(Prediction, Truth, alpha = n)) +
    geom_tile(show.legend = FALSE) +
    geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)


predictions_glm %>%
    metrics(Porosity, .pred_class) %>%
    select(-.estimator) %>%
    filter(.metric == "accuracy") %>%
    kable()


tibble(
    "precision" = 
        precision(predictions_glm, Porosity, .pred_class) %>%
        select(.estimate),
    "recall" = 
        recall(predictions_glm, Porosity, .pred_class) %>%
        select(.estimate)
) %>%
    unnest(cols = c(precision, recall)) %>%
    kable()


predictions_glm %>%
    f_meas(Porosity, .pred_class) %>%
    select(-.estimator) %>%
    kable()


predictions_glm %>%
    conf_mat(Porosity, .pred_class) %>%
    summary() %>%
    select(-.estimator) %>%
    filter(.metric %in% c("accuracy", "precision", "recall", "f_meas")) %>%
    kable()

cross_val_tbl <- vfold_cv(train_tbl, v = 10)
cross_val_tbl


cross_val_tbl %>% pluck("splits", 1)


#Random Forest
#
#

###
rf_fun <- function(split, id, try, tree) {
    
    analysis_set <- split %>% analysis()
    analysis_prepped <- analysis_set %>% recipe_simple()
    analysis_baked <- analysis_prepped %>% bake(new_data = analysis_set)
    
    model_rf <-
        rand_forest(
            mode = "classification",
            mtry = try,
            trees = tree
        ) %>%
        set_engine("ranger",
                   importance = "impurity"
        ) %>%
        fit(Porosity ~ ., data = analysis_baked)
    
    assessment_set     <- split %>% assessment()
    assessment_prepped <- assessment_set %>% recipe_simple()
    assessment_baked   <- assessment_prepped %>% bake(new_data = assessment_set)
    
    tibble(
        "id" = id,
        "truth" = assessment_baked$Porosity,
        "prediction" = model_rf %>%
            predict(new_data = assessment_baked) %>%
            unlist()
    )
    
}

pred_rf <- map2_df(
    .x = cross_val_tbl$splits,
    .y = cross_val_tbl$id,
    ~ rf_fun(split = .x, id = .y, try = 3, tree = 200)
)




tail(pred_rf,15)


pred_rf %>%
    conf_mat(truth, prediction) %>%
    summary() %>%
    select(-.estimator) %>%
    filter(.metric %in% c("accuracy", "precision", "recall", "f_meas")) %>%
    kable()

pred_rf %>%
    conf_mat(truth, prediction) %>%
    pluck(1) %>%
    as_tibble() %>%

# Visualize with ggplot
ggplot(aes(Prediction, Truth, alpha = n)) +
    geom_tile(show.legend = FALSE) +
    geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)
