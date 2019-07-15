diab= read.csv("diabetes.csv", header=T, stringsAsFactors=F)
db = read.csv('diabetes.csv', header=TRUE)
View(diab)

diab$Outcome <- as.factor(diab$Outcome)
levels(diab$Outcome) <- c("No","Yes")

nrows <- NROW(diab)
set.seed(69)				            # fix random value
index <- sample(1:nrows, 0.75 * nrows)	# shuffle and divide
# train <- diab                         # 768 test data (100%)
train <- diab[index,]			        # 576 test data (75%)
test <- diab[-index,]  		            # 192 test data (25%)
prop.table(table(train$Outcome))
library(caret)

#rpart
library(rpart)
learn_rp <- rpart(Outcome~.,data=train,control=rpart.control(minsplit=2))
pre_rp <- predict(learn_rp, test[,-9], type="class")
cm_rp  <- confusionMatrix(pre_rp, test$Outcome)	
cm_rp
plot(learn_rp, margin = 0.05); text(learn_rp, use.n = TRUE, cex = 0.6)

#rforest
library(randomForest)
learn_rf <- randomForest(Outcome~., data=train, ntree=500, proximity=T, importance=T)
pre_rf<- predict(learn_rf, test[,-9])
cm_rf<- confusionMatrix(pre_rf, test$Outcome)
cm_rf
plot(learn_rf, main="Random Forest (Error Rate vs. Number of Trees)",lwd=3)

#prune
learn_pru <- prune(learn_rp, cp=learn_rp$cptable[which.min(learn_rp$cptable[,"xerror"]),"CP"])
pre_pru <- predict(learn_pru, test[,-9], type="class")
cm_pru <-confusionMatrix(pre_pru, test$Outcome)			
cm_pru
plot(learn_pru, margin = 0.05); text(learn_pru, use.n = TRUE, cex = 0.7)

#LogisticRegression
table(db$Outcome) #baseline accuracy
set.seed(420)
require(caTools)
sample = sample.split(db$Outcome, SplitRatio=0.75)
train1 = subset(db, sample==TRUE)
test1 = subset(db, sample==FALSE)
# Fit model - using all independent variables
AllVar <- glm(Outcome ~ ., data = train1, family = binomial)
summary(AllVar)
# Let's predict outcome on Training dataset
PredictTrain <- predict(AllVar, type = "response")
summary(PredictTrain)

# Build confusion matrix with a threshold value of 0.5
threshold_0.5 <- table(train1$Outcome, PredictTrain > 0.5)
threshold_0.5
# Accuracy
accuracy_0.5 <- round(sum(diag(threshold_0.5))/sum(threshold_0.5),2)
sprintf("Accuracy is %s",accuracy_0.5)
# Mis-classification error rate
MC_0.5 <- 1-accuracy_0.5
sprintf("Mis-classification error is %s",MC_0.5)

# Build confusion matrix with a threshold value of 0.7
threshold_0.7 <- table(train1$Outcome, PredictTrain > 0.7)
threshold_0.7
# Accuracy
accuracy_0.7 <- round(sum(diag(threshold_0.7))/sum(threshold_0.7),2)
sprintf('Accuracy is %s', accuracy_0.7)
# Mis-classification error rate
MC_0.7 <- 1-accuracy_0.7
sprintf("Mis-classification error is %s",MC_0.7)

# Build confusion matrix with a threshold value of 0.3
threshold_0.3 <- table(train1$Outcome, PredictTrain > 0.3)
threshold_0.3
# Accuracy
accuracy_0.3 <- round(sum(diag(threshold_0.3))/sum(threshold_0.3),2)
sprintf("Accuracy is %s", accuracy_0.3)
# Mis-classification error rate
MC_0.3 <- 1-accuracy_0.3
sprintf("Mis-classification error is %s",MC_0.3)

# Generate ROC Curves
library(ROCR)
ROCRpred = prediction(PredictTrain, train1$Outcome)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Adding threshold labels
plot(ROCRperf, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
abline(a=0, b=1)
auc_train <- round(as.numeric(performance(ROCRpred, "auc")@y.values),2)
legend(.8, .2, auc_train, title = "AUC", cex=1)

# Making predictions on test set
PredictTest <- predict(AllVar, type = "response", newdata = test1)
# Convert probabilities to values using the below
## Based on ROC curve above, selected a threshold of 0.5
test_tab <- table(test1$Outcome, PredictTest > 0.5)
test_tab
accuracy_test <- round(sum(diag(test_tab))/sum(test_tab),2)
sprintf("Accuracy on test set is %s", accuracy_test)

# Compute test set AUC
ROCRPredTest = prediction(PredictTest, test1$Outcome)
auc = round(as.numeric(performance(ROCRPredTest, "auc")@y.values),2)
auc
