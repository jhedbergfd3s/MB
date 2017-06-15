
#Load packages
library(sqldf)
library(ggplot2)
library(plotly)
library(data.table)
library(mltools)
library(h2o)
library(dplyr)



#Set working directory
setwd("F:/kaggle/Mercedes-Benz")


#Load data files
sample_submission <- fread("data/sample_submission.csv")
train <- fread("data/train.csv")
test <- fread("data/test.csv")


#Add Train 0/1 Column to both train ans test datasets before merge
train$Train <- 1
test$Train <- 0


#Combine Train and Test datasets
train_test <- train
train_test$y <- NULL #remove response variable before rbind of two datasets
train_test <- rbind(train_test,test)


#### Feature Engineering Section
#One Hot Encoding for X0-X8
train_test <- as.data.frame(train_test)  #convert to data.frame for easier factorization
for(i in seq.int(2,9,1)){
    train_test[,i] <- as.factor(train_test[,i])
}

train_test <- as.data.table(train_test) #conver back to data.table for easier One hot Encoding
train_test <- one_hot(train_test,cols=colnames(train_test[,2:9, with=FALSE]),dropCols = FALSE)


#Identify duplicate columns in Training dataset, Remove duplicate columns from both
NonDuplicateFields <- colnames(train_test[, !duplicated(t(train_test)), with=FALSE])
NonDuplicateFields <- NonDuplicateFields[-2] #remove response variable
train_test <- train_test[, NonDuplicateFields, with=FALSE]


#Define R2 Function
R2 <- function(predicted, actual) {
  #sample_data_for_r2 <- data.frame(predicted=c(2,6.7143,11.4286,16.1429,20.8572,25.5715,30.2858),actual=c(8,8,8,9,15,30,35))
  y_bar <- mean(actual)
  yi_minus_ybar_squared <- c()
  yi_minus_yhat_squared <- c()
  for(i in seq.int(1,length(predicted),1)){
    yi_minus_ybar_squared[i] <- (actual[i]-y_bar)^2 
    yi_minus_yhat_squared[i] <- (actual[i]-predicted[i])^2
  }
  SStot <- sum(yi_minus_ybar_squared)
  SSres <- sum(yi_minus_yhat_squared)
  R2 <- 1-(SSres/SStot)
  rm(y_bar,yi_minus_ybar_squared,yi_minus_yhat_squared,i,SStot,SSres)
  return(R2)
  #R2(sample_data_for_r2$predicted,sample_data_for_r2$actual) #answer should be 0.77897
}





#Split train_test dataset into train_model and valid_model for model creation, also add back response variable
train_valid_w_features <- train_test[Train==1,]
train_valid_w_features <- cbind(y=train$y,train_valid_w_features)

      # 
      # #Looking to understand high values only
      # train_valid_w_features <- train_valid_w_features[y>=120]
#Create New Field if X0=(AZ,BC)
train_valid_w_features$Jeff_1 <- pmax(train_valid_w_features[,X0_az],train_valid_w_features[,X0_bc])



set.seed(12345)
sample_size <- round(0.8*dim(train_valid_w_features)[1],0)
train_ind <- sample(train_valid_w_features$ID, sample_size)
train_model <- train_valid_w_features[(ID %in% train_ind),]
valid_model <- train_valid_w_features[!(ID %in% train_ind),]




####Modeling Section
# h2o.init(nthreads = 6, max_mem_size = '10G') # uses 6 machine cores, 10G of RAM 
# h2o.shutdown(prompt = FALSE)
# Monitor at http://localhost:54321/flow/index.html

train_model_h2o <- as.h2o(train_model, destination_frame = "train_model.hex" )
valid_model_h2o <- as.h2o(valid_model, destination_frame = "valid_model.hex" )


y <- "y"
x <- colnames(train_model[,2:528, with=FALSE])
gbm_model_3 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
                       validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
                       x=x,                 ## this can be names or column numbers
                       y=y,                   ## target: using the logged variable created earlier
                       model_id="gbm_model_3",              ## internal H2O name for model
                       ntrees = 200,                  ## use fewer trees than default (50) to speed up training
                       learn_rate = 0.05,             ## lower learn_rate is better, but use high rate to offset few trees
                       max_depth = 3,
                       score_tree_interval = 1,      ## score every 3 trees
                       sample_rate = 0.5,            ## use half the rows each scoring round
                       col_sample_rate = 0.8,        ## use 4/5 the columns to decide each split decision
                       stopping_rounds = 5,
                       stopping_metric = "deviance", # "AUTO", "deviance", "logloss", "MSE", "RMSE", "MAE", "RMSLE", "AUC", "lift_top_group", "misclassification", "mean_per_class_error"
                       distribution = "gaussian", # "AUTO","bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie","laplace", "quantile", "huber")
                       categorical_encoding = "Eigen" #"AUTO" "Enum" "OneHotInternal" "OneHotExplicit" "Binary" "Eigen" 

)
#summary(gbm_model_3)
(h2o.r2(gbm_model_3,train=T)) # 0.595226  
(h2o.r2(gbm_model_3,valid=T)) # 0.5557255  
#h2o.saveModel(gbm_model_3,"gbm_model_3")
#h2o.performance(gbm_model_3, train_model_h2o)
#h2o.performance(gbm_model_3, valid_model_h2o)
#path <- getwd()
#h2o.download_mojo(gbm_model_3, path=path)
# java -cp h2o.jar hex.genmodel.tools.PrintMojo --tree 0 --detail --levels 25 -i gbm_model_3.zip -o gbm_model_3.gv
# dot -Tpng gbm_model_3.gv -o gbm_model_3.png


y <- "y"
x <- colnames(train_model[,2:528, with=FALSE])
rf_model_3 <- h2o.randomForest(training_frame = train_model_h2o,      ## H2O frame holding the training data
                               validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
                               x=x,                 ## this can be names or column numbers
                               y=y,                   ## target: using the logged variable created earlier
                               model_id="rf_model_3",              ## internal H2O name for model
                               ntrees = 2000,                  ## use fewer trees than default (50) to speed up training
                               #learn_rate = 0.005,             ## lower learn_rate is better, but use high rate to offset few trees
                               max_depth = 5,
                               score_tree_interval = 5,      ## score every 3 trees
                               sample_rate = 0.8,            ## use half the rows each scoring round
                               binomial_double_trees = TRUE,
                               stopping_rounds = 5
)
#summary(rf_model_3)
(h2o.r2(rf_model_3,train=T)) # 0.5680936
(h2o.r2(rf_model_3,valid=T)) # 0.5589466
#h2o.saveModel(rf_model_3,"rf_model_3")
#path <- getwd()
#h2o.download_mojo(rf_model_3, path=path)


gbm_model_3_valid_pred_y <- as.data.frame(h2o.predict(gbm_model_3,valid_model_h2o))[,1]
rf_model_3_valid_pred_y <- as.data.frame(h2o.predict(rf_model_3,valid_model_h2o))[,1]
    #lm_model_3 <- h2o.getModel('glm-ba53ef32-bad7-46e3-89ae-5ba03aefba28')
    #lm_model_3_valid_pred_y <- as.data.frame(h2o.predict(lm_model_3,valid_model_h2o))[,1]

R2(gbm_model_3_valid_pred_y,valid_model$y) # 0.5542835
R2(rf_model_3_valid_pred_y,valid_model$y) # 0.5551162
for(i in seq.int(0,10,1)){
  gbm_prct <- i/10
  rf_prct <- 1-gbm_prct
  print(paste('i=',i,'    ','gbm_prct=',gbm_prct*100,'    ','rf_prct=',rf_prct*100,'    ','R2=',R2(gbm_prct*gbm_model_3_valid_pred_y+rf_prct*rf_model_3_valid_pred_y,valid_model$y),sep=''))
}



valid_model$gbm_model_3 <- as.data.frame(h2o.predict(gbm_model_3,valid_model_h2o))[,1]
valid_model$rf_model_3 <- as.data.frame(h2o.predict(rf_model_3,valid_model_h2o))[,1]
valid_model$Ensemble_model_3 <- 0.4*valid_model$gbm_model_3 + 0.6*valid_model$rf_model_3
#valid_model$lm_model_3 <- as.data.frame(h2o.predict(lm_model_3,valid_model_h2o))[,1]

plot_ly(x=valid_model$y,y=valid_model$gbm_model_3,type="scatter", mode="markers") %>% add_trace(x=seq.int(80,120,1),y=seq.int(80,120,1), mode='lines') %>% layout(xaxis=list(title="Actual"),yaxis=list(title="Predicted-gbm_model_3"))
plot_ly(x=valid_model$y,y=valid_model$rf_model_3,type="scatter", mode="markers") %>% add_trace(x=seq.int(80,120,1),y=seq.int(80,120,1), mode='lines') %>% layout(xaxis=list(title="Actual"),yaxis=list(title="Predicted-rf_model_3"))
plot_ly(x=valid_model$y,y=valid_model$Ensemble_model_3,type="scatter", mode="markers") %>% add_trace(x=seq.int(80,120,1),y=seq.int(80,120,1), mode='lines') %>% layout(xaxis=list(title="Actual"),yaxis=list(title="Predicted-Ensemble_model_3"))
plot_ly(x=valid_model$y,y=valid_model$lm_model_3,type="scatter", mode="markers") %>% layout(xaxis=list(title="Actual"),yaxis=list(title="Predicted-lm_model_3"))


R2(gbm_model_3_valid_pred_y,valid_model$y) # 0.5557255
R2(rf_model_3_valid_pred_y,valid_model$y) # 0.5589466
R2(valid_model$Ensemble_model_3,valid_model$y) # 0.5604939


#Check for scaling improvements
scale_multiplier_i <- c()
scale_results_i <- c()
for(i in seq.int(.99,1.01,0.0005)){
  print(paste('i=',i,'    ','multplier=',i,'    ','R2=',R2(i*valid_model$Ensemble_model_3,valid_model$y),sep=''))
  scale_multiplier_i <- c(scale_multiplier_i,i)
  scale_results_i <- c(scale_results_i,R2(i*valid_model$Ensemble_model_3,valid_model$y))
}
scale_results <- data.frame(scale_multiplier_i=scale_multiplier_i,scale_results_i=scale_results_i)


valid_model$Ensemble_model_3 <- 0.999*valid_model$Ensemble_model_3
R2(valid_model$Ensemble_model_3,valid_model$y) # 0.5605639


# 
#     R2(gbm_model_3_valid_pred_y,valid_model$y) # 0.5542835
#     R2(lm_model_3_valid_pred_y,valid_model$y) # 0.5551162
#     for(i in seq.int(0,10,1)){
#       gbm_prct <- i/10
#       lm_prct <- 1-gbm_prct
#       print(paste('i=',i,'    ','gbm_prct=',gbm_prct*100,'    ','lm_prct=',lm_prct*100,'    ','R2=',R2(gbm_prct*gbm_model_3_valid_pred_y+lm_prct*lm_model_3_valid_pred_y,valid_model$y),sep=''))
#     }
# 
#     R2(lm_model_3_valid_pred_y,valid_model$y) # 0.5542835
#     R2(rf_model_3_valid_pred_y,valid_model$y) # 0.5551162
#     for(i in seq.int(0,10,1)){
#       lm_prct <- i/10
#       rf_prct <- 1-lm_prct
#       print(paste('i=',i,'    ','lm_prct=',lm_prct*100,'    ','rf_prct=',rf_prct*100,'    ','R2=',R2(lm_prct*lm_model_3_valid_pred_y+rf_prct*rf_model_3_valid_pred_y,valid_model$y),sep=''))
#     }




# 
# x <- colnames(train_model[,3:519, with=FALSE])
# pca_model_3 <- h2o.prcomp(
#                   train_model_h2o, 
#                   x, 
#                   model_id = "pca_model_3", 
#                   validation_frame = valid_model_h2o,
#                   ignore_const_cols = TRUE, 
#                   score_each_iteration = FALSE,
#                   transform = "NONE", #c("NONE", "STANDARDIZE", "NORMALIZE", "DEMEAN", "DESCALE"),
#                   pca_method = "GramSVD", #c("GramSVD", "Power", "Randomized", "GLRM")
#                   k = 10,
#                   max_iterations = 1000, 
#                   use_all_factor_levels = TRUE,
#                   compute_metrics = TRUE, 
#                   impute_missing = FALSE, 
#                   seed = -1,
#                   max_runtime_secs = 0)
# 
# 
# 
# temp <- as.data.frame(h2o.predict(pca_model_3,train_model_h2o))
# temp2 <- as.data.frame(h2o.predict(pca_model_3,valid_model_h2o))
# 
# train_model <- cbind(train_model,temp)
# valid_model <- cbind(valid_model,temp2)



# View(valid_model[,583:597])




      #REFERENCE - gbm_model_1
      # y <- "y"
      # x <- colnames(train_model[,2:379, with=FALSE])
      # gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
      #                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
      #                        x=x,                 ## this can be names or column numbers
      #                        y=y,                   ## target: using the logged variable created earlier
      #                        model_id="gbm_model_1",              ## internal H2O name for model
      #                        ntrees = 900,                  ## use fewer trees than default (50) to speed up training
      #                        learn_rate = 0.004,             ## lower learn_rate is better, but use high rate to offset few trees
      #                        max_depth = 3,
      #                        score_tree_interval = 15,      ## score every 3 trees
      #                        sample_rate = 0.5,            ## use half the rows each scoring round
      #                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
      # )
      # summary(gbm_model_1)
      # (h2o.r2(gbm_model_1,train=T)) # 0.5886679
      # (h2o.r2(gbm_model_1,valid=T)) # 0.553584
      # #h2o.saveModel(gbm_model_1,"gbm_model_1")
      # h2o.performance(gbm_model_1, train_model_h2o)
      # h2o.performance(gbm_model_1, valid_model_h2o)



#Score Test Data
test_h2o <- as.h2o(test, destination_frame = "test.hex")
h2o.loadModel("F:\\kaggle\\Mercedes-Benz\\gbm_model_3\\gbm_model_3") #gbm_model_3
gbm_model_3 <- h2o.getModel("gbm_model_3")
test_pred_y_gbm_model_3 <- as.data.frame(h2o.predict(gbm_model_3,test_h2o))[,1]

h2o.loadModel("F:\\kaggle\\Mercedes-Benz\\rf_model_3\\rf_model_3") #rf_model_3
rf_model_3 <- h2o.getModel("rf_model_3")
test_pred_y_rf_model_3 <- as.data.frame(h2o.predict(rf_model_3,test_h2o))[,1]


test_pred_y <- 0.999*( 0.4*test_pred_y_gbm_model_3 + 0.6*test_pred_y_rf_model_3)



#Create Submission File
submission_name <- "MB_JMH_Submission_3"  #Only need to change in this one place in future code versions
assign(paste(submission_name,sep=''), data.frame(ID=test$ID,y=test_pred_y))

#Check Submission File for NULLs and generate summary stats
length(get(submission_name)$y[is.na(get(submission_name)$y)]) #0
min(get(submission_name)$y) # 77.5437
max(get(submission_name)$y) # 118.4956
mean(get(submission_name)$y) # 100.7939

#Output Submission File
write.csv(get(submission_name),paste(submission_name,'.csv',sep=''), row.names = FALSE)

#Leaderboard score = 0.55736
