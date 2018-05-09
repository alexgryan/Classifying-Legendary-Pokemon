library(caret)
library(doSNOW)
#Loading Data
train = read.csv("Pokemon.csv", stringsAsFactors = FALSE)
View(train)

#Data Manipulation and Feature Engineering

#Filling in blanks for Pokemon's second type
table(train$Type.2)
train$Type.2[train$Type.2 == ""] <- "No Second Type"

#Create Features to Distinguish Legendaries from non-Legendaries
train$OverallStrength <- ifelse(train$Total > median(train$Total),
                                "Strong", "Weak")
train$isDragon <- ifelse(train$Type.1 == "Dragon", "Y", "N")
train$hasHighHP <- ifelse(train$HP > median(train$HP), "Y", "N")
train$hasHighAttack <- ifelse(train$Attack > median(train$Attack), "Y", "N")
train$hasHighDefense <- ifelse(train$Defense > median(train$Defense), "Y", "N")
train$hasHighSpAtk <- ifelse(train$Sp..Atk > median(train$Sp..Atk), "Y", "N")
train$hasHighSpDef <- ifelse(train$Sp..Def > median(train$Sp..Def), "Y", "N")
train$hasHighSpeed <- ifelse(train$Speed > median(train$Speed), "Y", "N")

#Factoring Categorical Data
train$Type.1 <- as.factor(train$Type.1)
train$Type.2 <- as.factor(train$Type.2)
train$Generation <- as.factor(train$Generation)
train$Legendary <- as.factor(train$Legendary)
train$OverallStrength <- as.factor(train$OverallStrength)
train$isDragon <- as.factor(train$isDragon)
train$hasHighHP <- as.factor(train$hasHighHP)
train$hasHighAttack <- as.factor(train$hasHighAttack)
train$hasHighDefense <- as.factor(train$hasHighDefense)
train$hasHighSpAtk <- as.factor(train$hasHighSpAtk)
train$hasHighSpDef <- as.factor(train$hasHighSpDef)
train$hasHighSpeed <- as.factor(train$hasHighSpeed)

View(train)
#Subsets the data we have factored and created features for
features <- c("Type.1", "Type.2", "Generation", "Legendary","OverallStrength", "isDragon", "hasHighHP", "hasHighDefense","hasHighSpAtk","hasHighSpDef","hasHighSpeed")
train <- train[, features]
str(train)

#Creating the Training and Test Splits
set.seed(54234)
indexes <- createDataPartition(train$Legendary,
                               times = 1,
                               p = 0.75,
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

#Gather training model with 2 clusters to identify legendary pokemon
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

#Creating the Training and Test Splits for Dragon Classifier
set.seed(54234)
indexes <- createDataPartition(train$isDragon,
                               times = 1,
                               p = 0.75,
                               list = FALSE)
dragon.train <- train[indexes,]
dragon.test <- train[-indexes,]


# Examine the proportions of the Survived class lable across
# the datasets.
prop.table(table(train$isDragon))
prop.table(table(dragon.train$isDragon))
prop.table(table(dragon.test$isDragon))

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

#Creating Cluster
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Gather training model with 2 clusters to identify legendary pokemon
caret.cv <- train(isDragon ~ ., 
                  data = dragon.train,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)

caret.cv

#Naive Bayes Classifier
library(e1071)

NB = naiveBayes(pokemon.train, pokemon.train$Legendary,laplace = 1)
nbHat = predict(NB,pokemon.test$Legendary)
confusionMatrix(nbHat, pokemon.test$Legendary)

#KNN Classification
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

knn_fit <- train(Legendary ~., data = pokemon.train, method = "knn",
                 trControl= train.control,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

stopCluster(cl)

knn_fit

knnHat = predict(knn_fit, pokemon.test)
confusionMatrix(knnHat, pokemon.test$Legendary)
ctable <- as.table(confusionMatrix(knnHat, pokemon.test$Legendary), nrow = 2, byrow = TRUE)
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")
