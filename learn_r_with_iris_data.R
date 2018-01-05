##Run line by line to understand how the code works

#Install these packages prior to running
#Then load packages
require("dplyr")
require("tidyr")
require("magrittr")
require("ggplot2")
require("caret")
require("e1071")
library(datasets)

##load the iris dataset as a dataframe for easier viewing
iris <- as.data.frame(iris)
View(iris)

#summarize the dataset
sum <- as.data.frame(summary(iris))
View(sum)

#visualize the whole dataset
pairs(iris)
#color code the visual
pairs(iris, col= iris$Species)

##COMMENT THIS SECTION
panel.pearson <- function(x, y, ...) {
  horizontal <- (par("usr")[1] + par("usr")[2]) / 2; 
  vertical <- (par("usr")[3] + par("usr")[4]) / 2; 
  text(horizontal, vertical, format(abs(cor(x,y)), digits=2)) 
}
pairs(iris[1:4], main = "Data Set Correlations", pch = 21, bg = c("red","green3","blue")[unclass(iris$Species)], upper.panel=panel.pearson)


#set up 4x4 plot
par(mfrow=c(2,2))
#add histograms for each variable
hist(iris$Sepal.Length, xlab = "Sepal Length", main = "Histogram of Sepal Length")
hist(iris$Sepal.Width, xlab = "Sepal Width", main = "Histogram of Sepal Width")
hist(iris$Petal.Length, xlab = "Petal Length", main = "Histogram of Petal Length")
hist(iris$Petal.Width, xlab = "Petal Width", main = "Histogram of Petal Width")

#set up 1x1 plot
par(mfrow=c(1,1))

#find where the dataframes contains data for the setosa and versicolor species (virginica not included)
is <- iris$Species == "setosa"
iv <- iris$Species == "versicolor"

##dimensions of a setosa petal
is_len= iris[is, c(1,3)]
##dimensions of a setosa sepal
is_wid= iris[is, c(2,4)]
##dimensions of a versicolor petal
iv_len= iris[iv, c(1,3)]
##dimensions of a versicolor sepal
iv_wid= iris[iv, c(2,4)]

##set axes limits and chart labels
matplot(c(0, 8), c(0, 6), type =  "n", xlab = "Length", ylab = "Width",
        main = "Petal and Sepal Dimensions in Iris Blossoms", sub= "S= Setosa, V=Versicolor, red= petal, black= sepal")

##plot all 4 data sets, points for sepal will show up as a letter s and points for versicolor will show up as letter v
##petal dimensions are plotted in red, sepal dimensions are plotted in black
matpoints(is_len, is_wid, pch = "SS")
matpoints(iv_len, iv_wid, pch = "VV")

##plot the sepal dimensions of each species
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_jitter() +
  facet_grid(. ~ Species)

##plot the petal dimensions of each species
ggplot(iris, aes(x = Petal.Length, y = Petal.Width)) +
  geom_jitter() +
  facet_grid(. ~ Species)

##See https://www.kaggle.com/antoniolopez/iris-data-visualization-with-r for additional visualization tutorials


##REGRESSION analysis using only petal length and seppal length
##Create a second iris element called Is.Versicolor with just the index where the species is versicolor
iris[['Is.Versicolor']] <- as.numeric(iris[['Species']] == 'versicolor')

##Try using a linear model to carry out regression
fit.lm <- lm(Is.Versicolor ~ Petal.Length + Sepal.Length, data = iris)

##Print a summary of the regression analysis and all statistics
summary(fit.lm)

##Create a third iris element which gives prediction values for each row of values with the regression equation created
pred <- predict(fit.lm)

##comvert this to a binary indicator (is versicolor > 0.5 and is not versicolor < 0.5)
iris[['Predict.Versicolor.lm']] <- as.numeric(pred > 0.5)

## compare actual value and predicted value binary
fit.lm.outcome <- table(iris[, c('Is.Versicolor', 'Predict.Versicolor.lm')])


##LOGISTIC REGRESSION analysis using only petal length and seppal length
fit.logit <- glm(Is.Versicolor ~ Petal.Length + Sepal.Length, data = iris, family = binomial(link = 'logit'))
summary(fit.logit)
iris[['Predict.Versicolor.logit']] <- as.numeric(predict(fit.logit) > 0.5)
fit.logit.outcome <- table(iris[, c('Is.Versicolor', 'Predict.Versicolor.logit')])


##REGRESSION analysis using only petal length and seppal length
all.lm <- lm(Is.Versicolor ~ Petal.Length + Sepal.Length+ Petal.Width + Sepal.Width, data = iris)
summary(fit.lm)
pred <- predict(fit.lm)
iris[['Predict.Versicolor.lm']] <- as.numeric(pred > 0.5)
all.lm.outcome <- table(iris[, c('Is.Versicolor', 'Predict.Versicolor.lm')])


##LOGISTIC REGRESSION analysis using all 4 variables
fit.logit <- glm(Is.Versicolor ~ Petal.Length + Sepal.Length + Petal.Width + Sepal.Width, data = iris, family = binomial(link = 'logit'))
summary(fit.logit)
iris[['Predict.Versicolor.logit']] <- as.numeric(predict(fit.logit) > 0.5)
all.logit.outcome <- table(iris[, c('Is.Versicolor', 'Predict.Versicolor.logit')])


##CARET machine learning
#partition the data into testing and training sets (75:25 training:test)
index <- createDataPartition(iris$Species, p=0.75, list=FALSE)
iris.training <- iris[index,]
iris.test <- iris[-index,]

#see all the methods availale to the caret package
names(getModelInfo())
#Train model with different methods, columns 1-4 are the features (RHS) and column 5 is the target variable (LHS)
model_knn <- train(iris.training[, 1:4], iris.training[, 5], method='knn')
model_cart <- train(iris.training[, 1:4], iris.training[, 5], method='rpart2')
# Predict the labels of the test set
prediction_knn<-predict(object=model_knn,iris.test[,1:4])
prediction_cart<-predict(object=model_cart,iris.test[,1:4])
# Evaluate the predictions
table(prediction_knn)
table(prediction_cart)
# Confusion matrix 
confusionMatrix(prediction_knn,iris.test[,5])
confusionMatrix(prediction_cart,iris.test[,5])

## Train model with preprocessing such as centering and scaling
model_knn <- train(iris.training[, 1:4], iris.training[, 5], method='knn', preProcess=c("center", "scale"))
predictions<-predict.train(object=model_knn,iris.test[,1:4], type="raw")
confusionMatrix(predictions,iris.test[,5])
