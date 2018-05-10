library(caret)
library(doSNOW)
library(magrittr)
library(dplyr)
library(ggthemes)
library(fmsb)
setwd("D:/PokemonAnalytic/Classifying-Legendary-Pokemon")
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
train$isPsychic <- ifelse(train$Type.1 == "Psychic", "Y", "N")
train$isFlying <- ifelse(train$Type.1 == "Flying", "Y", "N")

#Factoring Categorical Data
train$Type.1 <- as.factor(train$Type.1)
train$Type.2 <- as.factor(train$Type.2)
train$Generation <- as.factor(train$Generation)
train$Legendary <- as.factor(train$Legendary)
train$OverallStrength <- as.factor(train$OverallStrength)
train$isDragon <- as.factor(train$isDragon)
train$isPsychic <- as.factor(train$isPsychic)
train$isFlying <- as.factor(train$isFlying)

View(train)

#Visualization for Legendaries and Non-Legendaries by Type
pokemon<-read.csv("Pokemon.csv",sep=",",stringsAsFactors=F)
colnames(pokemon)<-c("id","Name","Type.1","Type.2", "Total", "HP","Attack","Defense","Sp.Atk","Sp.Def","Speed","Generation","Legendary")
Type.1<-c("Dragon","Steel","Flying","Psychic","Rock" ,"Fire","Electric" ,"Dark","Ghost" ,"Ground","Ice", "Water","Grass","Fighting", "Fairy" ,"Poison","Normal","Bug")
color<-c("#6F35FC","#B7B7CE","#A98FF3","#F95587","#B6A136","#EE8130","#F7D02C","#705746","#735797","#E2BF65","#96D9D6","#6390F0","#7AC74C","#C22E28","#D685AD","#A33EA1","#A8A77A","#A6B91A")
COL<-data.frame(Type.1,color)

merge(
  merge(pokemon %>% dplyr::group_by(Type.1) %>% dplyr::summarize(tot=n()),
        pokemon %>% dplyr::group_by(Type.1,Legendary) %>% dplyr::summarize(count=n()),by='Type.1'),
  COL, by='Type.1') %>% 
  ggplot(aes(x=reorder(Type.1,tot),y=count)) + 
  geom_bar(aes(fill=color,alpha=Legendary),color='white',size=.25,stat='identity') + 
  scale_fill_identity() + coord_flip() + theme_fivethirtyeight() + 
  ggtitle("Pokemon Distribution") + scale_alpha_discrete(range=c(.9,.6))

res<-data.frame(pokemon %>% dplyr::select(Type.1,HP, Attack, Defense, Sp.Atk, Sp.Def, Speed) %>% dplyr::group_by(Type.1) %>% dplyr::summarise_all(funs(mean)) %>% mutate(sumChars = HP + Attack + Defense + Sp.Atk + Sp.Def + Speed) %>% arrange(-sumChars))
res$color<-color
max<- ceiling(apply(res[,2:7], 2, function(x) max(x, na.rm = TRUE)) %>% sapply(as.double)) %>% as.vector
min<-rep.int(0,6)

par(mfrow=c(3,6))
par(mar=c(1,1,1,1))
for(i in 1:nrow(res)){
  curCol<-(col2rgb(as.character(res$color[i]))%>% as.integer())/255
  radarchart(rbind(max,min,res[i,2:7]),
             axistype=2 , 
             pcol=rgb(curCol[1],curCol[2],curCol[3], alpha = 1) ,
             pfcol=rgb(curCol[1],curCol[2],curCol[3],.5) ,
             plwd=2 , cglcol="grey", cglty=1, 
             axislabcol="black", caxislabels=seq(0,2000,5), cglwd=0.8, vlcex=0.8,
             title=as.character(res$Type.1[i]))
}

#Subsets the data we have factored and created features for
features <- c("Type.1", "Type.2", "Generation", "Legendary","OverallStrength", "isDragon", "isPsychic", "isFlying")
train <- train[, features]
str(train)

#Creating the Training and Test Splits
set.seed(54234)
indexes <- createDataPartition(train$Legendary,
                               times = 1,
                               p = 0.8,
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

tune.grid <- expand.grid(eta = c( 0.1, 0.2, 0.3, 0.4, 0.5),
                         nrounds = c(50, 75, 100),
                         max_depth = 4:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.5, 0.75, 1.0),
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

xgbHat = predict(caret.cv, pokemon.test)
confusionMatrix(xgbHat, pokemon.test$Legendary)

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
