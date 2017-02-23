# This programme uses a random forest to train the model for predicting scores and then we select the sentences with the maximum scores.

library(caret)
library(randomForest)
library(dplyr)

match_1<-read.csv("E:/final_data_football/match_1.csv")

match_1 <- match_changed_time(match_1)
train_full <- match_1
for (i in 3:30)
{
  match_i <- read.csv(paste("E:/final_data_football/match_",as.character(i),".csv",sep=""))
  match_i <- match_changed_time(match_i)
  train_full <- rbind(train_full,match_i)
}
train_full= train_full[complete.cases(train_full),] #removing NA Values
target = train_full$score

train_full_1 = train_full[,c(4:10,23)]


train_full$target = target
train_full_1$target = target


for (i in 1:nrow(train_full_1))
{
  train_full_1$sum[i] = sum(train_full[i,c(11,13:22)]) 
  
  
}
fit = randomForest(target~.,data= train_full_1,ntree = 500)


#Testing for match_1
match_1 <- read.csv("E:/final_data_football/match_1.csv")
match_1 = match_changed_time(match_1)
test_full = match_1[,c(4:10,23)]
for (i in 1:nrow(test_full))
{
  
  test_full$sum[i] = sum(match_1[i,c(11,13:22)]) 
  
}
pred = predict(fit, test_full, type = "response")
out_1<- cbind(as.character(match_1$text),match_1$pos,pred)
out_3<-data.frame(out_1)
out_3$pred = as.numeric(as.character(out_3$pred))
out_2 <- out_3[order(out_3$pred,decreasing = TRUE),]
out_4 <- out_2[1:30,]
out_4$V2 = as.numeric(as.character(out_4$V2))
out_4 = arrange(out_4,V2)
write.table(out_4$V1,"C:/football_summary_1.txt",row.names = FALSE,col.names = FALSE)


#Testing for rest of the matches
#Creating summaries for all matches
for(j in 3:30)
{
  match_j = read.csv(paste("E:/final_data_football/match_",as.character(j),".csv",sep=""))
  match_j = match_changed_time(match_j)
  test_j = match_j[,c(4:10,23)]
  for (i in 1:nrow(test_j))
  {
    
    test_j$sum[i] = sum(match_j[i,c(11,13:22)]) 
    
  }
  
  test_j= test_j[complete.cases(test_j),] #removing NA Values
  pred_j = predict(fit, test_j, type = "response")
  
  out_1<- cbind(as.character(match_j$text),match_j$pos,pred)
  out_3<-data.frame(out_1)
  out_3$pred = as.numeric(as.character(out_3$pred))
  out_2 <- out_3[order(out_3$pred,decreasing = TRUE),]
  out_4 <- out_2[1:30,]
  out_4$V2 = as.numeric(as.character(out_4$V2))
  out_4 = arrange(out_4,V2)
  write.table(out_4$V1,paste("C:/football_summary_",as.character(j),".txt",sep = "")
              ,row.names = FALSE,col.names = FALSE)
  
}
diff = abs(match_1$score - pred)
summary(diff)
varImpPlot(fit,main = "Feature Importance Random Forest ")
plot(fit,main = "Error Plot Random Forest")

importance(fit)
str(train)
