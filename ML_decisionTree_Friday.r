df <- Friday.WorkingHours.Afternoon.DDos.pcap_ISCX

library(mice)
?mice

set.seed(123)
mice(df)

table(df$Label)

set.seed(145)
trainIndex <- sample(1:nrow(df), size = 0.8*nrow(df))

trainSet <- df[trainIndex,]
testSet <- df[-trainIndex,]


nrow(trainSet)
nrow(testSet)
table(trainSet$Label)
table(testSet$Label)

type_sum(df$Label)

trainSet$Label <- as.factor(trainSet$Label)
testSet$Label <- as.factor(testSet$Label)

install.packages("rpart")
library(rpart)

modelEntropy <- rpart(Label ~ . , data = trainSet, method = "class" , 
                      parms= list(split = "information"))

modelEntropy2 <- rpart(Label ~ . , data = trainSet, method = "class" , 
                      parms= list(split = "information"))

modelGini <- rpart(Label ~ . , data = trainSet, method = "class" , 
                      parms= list(split = "gini"))

modelEntropy
modelEntropy2
modelGini

install.packages("rattle")
library(rattle)

fancyRpartPlot(modelEntropy)

summary(modelEntropy)
summary(modelGini)

modelEntropyHyper <- rpart(Label ~ . , data = trainSet, method = "class",
                           parms = list(split = "information") ,
                           control = rpart.control(minsplit = 20 , cp = 0.005 , maxdepth = 12))
modelEntropyHyper



predModelEntropy <- predict(modelEntropy , testSet , type = "class")
predModelGini <- predict(modelGini , testSet , type = "class")
predModelEntropyHyper <- predict(modelEntropyHyper , testSet , type="class")

library(caret)

confusionMatrix(predModelEntropy , testSet$Label)
confusionMatrix(predModelEntropy , testSet$Label , mode = "prec_recall")

printcp(modelEntropy)

confusionMatrix(predModelGini , testSet$Label)
confusionMatrix(predModelGini , testSet$Label , mode = "prec_recall")

confusionMatrix(predModelEntropyHyper , testSet$Label)
confusionMatrix(predModelEntropyHyper , testSet$Label , mode = "prec_recall")



##########---------- Neural Networks ----------###########

## 1. işlem: Data hazırlığı
modelData <- Friday.WorkingHours.Afternoon.DDos.pcap_ISCX
IndexOfBEING <- which(modelData$Label == "BENIGN")
modelData$Label[IndexOfBEING] <- 0 #BENIGN 0 ile gösterilecek
table(modelData$Label)
IndexOfDDos <- which(modelData$Label == "DDoS")
modelData$Label[IndexOfDDos] <- 1 #DDoS 1 ile gösterilecek
table(modelData$Label)
modelData$Label <- as.factor(modelData$Label)

##2. işlem: Scaling Data
#veri dağılımımız çok düzensiz bu düzensizlik NN'de sapmalara sebebiyet verebilir.
# bu yüzden verilerimizi ortalamaya göre scale edeceğiz.

modelScale <- preProcess(modelData, method = c('center', 'scale'))
modelDataScaled <- predict(modelScale, modelData)
colSums(is.na(modelDataScaled)) ##--> scale işlemi sonrası NA değer kontrolü


#--> modelScale ile gelen oranların predict ile birlikte verilerimze değer ataması yapıldı
View(modelDataScaled)

##3. işlem: trainset ve test set'in oluşturulması

set.seed(165)
trainIndex <- sample(1:nrow(modelDataScaled), size = 0.75*nrow(modelDataScaled))
trainSet <- modelDataScaled[trainIndex,] 
testSet <- modelDataScaled[-trainIndex,] 

nrow(trainSet)
table(trainSet$Label)
nrow(testSet)
table(testSet$Label)

###4. işlem: model eğitimi

?neuralnet
colSums(is.na(trainSet))
colSums(is.na(testSet))
trainSet$Flow.Bytes.s <- NULL
trainSet$Flow.Packets.s <- NULL

testSet$Flow.Bytes.s <- NULL
testSet$Flow.Packets.s <- NULL


trainSet$Label <- as.numeric(trainSet$Label)
testSet$Label <- as.numeric(testSet$Label)

modelNN_1 <- neuralnet(Label ~ . , data = trainSet,
                       hidden = 1, threshold = 0.01,
                       act.fct = 'logistic',
                       linear.output = FALSE
)


#### 5. işlem: Tahmin

# Model 1
predModel_1 <- predict(modelNN_1 , testSet)
predModel_1C <- ifelse(apply(predModel_1 , 1 , which.max) == 1 , "0" , "1")

library(caret)
confusionMatrix(as.factor(predModel_1C) , testSet$Label)

















