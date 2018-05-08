library(caret)
library(doSNOW)
#Loading Data
train = read.csv("Pokemon.csv", stringsAsFactors = FALSE)
View(train)

#Data Manipulation and Feature Engineering

#Filling in blanks for Pokemon's second type
table(train$Type.2)
train$Type.2[train$Type.2 == ""] <- "No Second Type"

#Factoring Categorical Data
train$Name <- as.factor(train$Name)
train$Type.1 <- as.factor(train$Type.1)
train$Type.2 <- as.factor(train$Type.2)
train$Generation <- as.factor(train$Generation)
train$Legendary <- as.factor(train$Legendary)

#Subsets the data we have factored and created features for
features <- c("Name", "Type.1", "Type.2", "Generation", "Legendary")
train <- train[, features]
str(train)

#Creating the Training and Test Splits
set.seed(542)
indexes <- createDataPartition(train$Legendary,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
pokemon.train <- train[indexes,]
pokemon.test <- train[-indexes,]


# Examine the proportions of the Survived class lable across
# the datasets.
prop.table(table(train$Legendary))
prop.table(table(pokemon.train$Legendary))
prop.table(table(pokemon.test$Legendary))

# Cross validation and tuning for model
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 4:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
View(tune.grid)

#Creating Cluster
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Gather training model with 2 clusters
caret.cv <- train(Legendary ~ ., 
                  data = pokemon.train,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)

caret.cv

#Predict based off tuning paramers
preds <- predict(caret.cv, pokemon.test)
confusionMatrix(preds, pokemon.test$Legendary)

#Naive Bayes Classifier
library(e1071)

NB = naiveBayes(pokemon.train, pokemon.train$Legendary,laplace = 1)
nbHat = predict(NB,pokemon.test$Legendary)
confusionMatrix(nbHat, pokemon.test$Legendary)
