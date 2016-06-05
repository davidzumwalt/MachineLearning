# Load in the package
library(randomForest)
library(ggplot2)
library(rpart)
library(caret)
library(e1071)


# Import the training set: train
train_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(train_url)

# Import the testing set: test
test_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(test_url)

# For the purposes of combining the data sets, we'll initialize a new feature 
# for the test set

test$Survived <- 0

# Note: RandomForest method requires there to be no NA values

# All data, both training and test set
all_data <- rbind(train, test)

# The names of the passengers includes each passenger's honorific.
# We can add a new feature called Title which contains these fields.
# First we have to parse the names and strip out the honorific.

#head(all_data$Name,10)

# We see that each name follows a LastName, Honorific. FirstName MiddleName format.
# Extract the honorific by first converting the Name column from a factor to a character vector.
all_data$Name <- as.character(all_data$Name)

# Go through the list of names in the data set and set the Title to be the honorific in the name.
all_data$Title <- sapply(all_data$Name, function(x) {gsub(" ","",unlist(strsplit(x, split='[,.]'))[2])})

# Now convert the Titles back into a new factor.

# all_data$Title <- factor(all_data$Title)

# summary(all_data$Title)

# We see that there are only a few people with the title Mlle, Mme, Don, Dona, Jonkheer,
# and a few other honorifics. We might do better if we group them instead with similar titles.

all_data$Title[all_data$Title %in% c('Mme','Mlle')] <- 'Mlle'
all_data$Title[all_data$Title %in% c('Capt','Don','Major','Sir')] <- 'Sir'
all_data$Title[all_data$Title %in% c('Dona','Jonkheer','Lady','theCountess')] <- 'Lady'

# Refactorize after replacement of Titles.
all_data$Title <- factor(all_data$Title)

# Passengers on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
all_data$Embarked[c(62, 830)] <- "S"

# Factorize embarkment codes.
all_data$Embarked <- factor(all_data$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
all_data$Fare[1044] <- median(all_data$Fare, na.rm = TRUE)

# How can we fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# We use method = "anova" since we are predicting a continuous variable.

predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked+Title,
                       data = all_data[!is.na(all_data$Age),], method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])

# Split the data back into a train set and a test set
train <- all_data[1:891,]
test <- all_data[892:1309,]

# For the purposes of cross validation, we'll take some of our labeled (training) data
# and test the robustness of our model by repeatedly sampling our training set to create
# a CV set from which to evaluate the model accuracy, precision, and recall.

# Set seed for reproducibility
set.seed(111)


# We only needed the Survived feature so it could agree with the training set when combining sets.
# We can remove it now.
test<-subset(test,select=-Survived)

# Apply the Random Forest Algorithm
#my_forest <- randomForest(as.factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title,data=train,importance=TRUE,ntree=1000)

TrainingControl <- trainControl(method = "repeatedcv",## 10-fold CV
                                number = 10, ## repeated ten times
                                repeats = 10)

# Optionally control the number of variables the folds have to include by adding
# the following field into train() - this takes a long time to run, however:
# tuneGrid = expand.grid(mtry = seq(2,12, by = 1))

my_forest <- train(as.factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title,data=train,
                       trControl = TrainingControl,
                       importance=TRUE,ntree=500)

my_forest$finalModel
#In proportion:
confusionMatrix(my_forest)

# Print the Accuracy and AccuracySD term for the best tune of your cross-validated forest
my_forest$results[my_forest$results$mtry==my_forest$bestTune[1,],]

# Make your prediction using the test set
my_prediction <- predict(my_forest, test, type="raw")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution_new.csv", row.names = FALSE)

#View which variables are most important to the model

imp <- importance(my_forest, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

importanceplot <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#2580E8") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("") +
  ylab("Feature Importance") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

importanceplot

#Clearly, one's Title and Class are the strongest predictors of survival rate. Sex is
# also quite important.

#Option to save Feature Importance plot
#ggsave("featureimportance.png", importanceplot)