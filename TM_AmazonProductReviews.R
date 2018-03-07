install.packages(c("tm","SnowballC","dplyr","wordcloud","rpart", "rpart.plot", "e1071", "nnet","pROC","ROCR","plyr","RWeka"))

library(tm)
library(SnowballC)
library(dplyr)
library(wordcloud)
library(rpart)
library(rpart.plot)
library(e1071)
library(nnet)
library(pROC)
library(ROCR)
library(plyr)
library(RWeka) # Note: If unable to install Rweka package, update JAVA by downloading and installing Windows Offline (64-bit)from https://www.java.com/en/download/manual.jsp

directory = "C:/Users/ctan/Documents/SEM2_SocialMediaAnalytics/Project/"
setwd(directory)
set.seed(1111) # To ensure aligned results with what is written in report.

######################################################################################################################## 
################################################### DATA PREPARATION ################################################### 
######################################################################################################################## 

# Parameter for sparsity
sparsity = 0.95

######################################### DICTIONARY #################################################

# Import precompiled list of negative and positive words
  # Source: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html (NOTE!!!: To send the files as attachment)
neg_words = read.table("negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("positive-words.txt", header = F, stringsAsFactors = F)[, 1]

#?neg_words = read.table("./negative-words.txt", header = F, stringsAsFactors = F)[, 1]
#?pos_words = read.table("./positive-words.txt", header = F, stringsAsFactors = F)[, 1]

# Additional words to add into dictionary - add NOT/NOR/NO/NEITHER to change negative to positive, and positive to negative words.
#   To change negative to positive words
neg_words_bi_not = paste("not",neg_words , sep = " ")
neg_words_bi_nor = paste("nor",neg_words , sep = " ")
neg_words_bi_no = paste("no",neg_words , sep = " ")
neg_words_bi_neither = paste("neither",neg_words , sep = " ")

pos_words_bi = c(pos_words, neg_words_bi_not, neg_words_bi_nor, neg_words_bi_no, neg_words_bi_neither)

#   To change positive to negative words
pos_words_bi_not = paste("not",pos_words , sep = " ")
pos_words_bi_nor = paste("nor",pos_words , sep = " ")
pos_words_bi_no = paste("no",pos_words , sep = " ")
pos_words_bi_neither = paste("neither",pos_words , sep = " ")

neg_words_bi = c(neg_words, pos_words_bi_not, pos_words_bi_nor, pos_words_bi_no, pos_words_bi_neither)

######################################### Load & Import Data #########################################

# Load all positive and negative datasets
review_corpus_pos = VCorpus(DirSource("./sentiment_amazon_product_reviews/positive")) 
review_corpus_neg = VCorpus(DirSource("./sentiment_amazon_product_reviews/negative")) 

######################################### PREPROCESSING ##############################################

# remove punctuation
review_corpus_pos = tm_map(review_corpus_pos, removePunctuation)
review_corpus_neg = tm_map(review_corpus_neg, removePunctuation)

# remove number
review_corpus_pos = tm_map(review_corpus_pos, removeNumbers)
review_corpus_neg = tm_map(review_corpus_neg, removeNumbers)

# strip white spaces
review_corpus_pos = tm_map(review_corpus_pos, stripWhitespace)
review_corpus_neg = tm_map(review_corpus_neg, stripWhitespace)

# transform to lowcase
review_corpus_pos = tm_map(review_corpus_pos, content_transformer(tolower))
review_corpus_neg = tm_map(review_corpus_neg, content_transformer(tolower))

# stopword removal
stopwords_list = c("the", "and", "ipod", "player", "dobrhope", "doand", stopwords("english")[-(80:98)])
review_corpus_pos = tm_map(review_corpus_pos, removeWords, stopwords_list) 
review_corpus_neg = tm_map(review_corpus_neg, removeWords, stopwords_list) 

# Stemming
review_corpus_pos = tm_map(review_corpus_pos, stemDocument)
review_corpus_neg = tm_map(review_corpus_neg, stemDocument)

# Use 1 and 2 GRAM 
# Create function to use one and bigram at the same time doing (min = 1, max = 2) 
BigramTokenizer = function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))

# Create DTM  
review_dtm_pos = DocumentTermMatrix(review_corpus_pos, control = list(tokenize = BigramTokenizer)) 
review_dtm_neg = DocumentTermMatrix(review_corpus_neg, control = list(tokenize = BigramTokenizer)) 

# Examine dtm with 1 and 2 gram included and remove Sparse terms
inspect(review_dtm_pos)
inspect(review_dtm_neg)
review_dtm_pos = removeSparseTerms(review_dtm_pos, sparsity)
review_dtm_neg = removeSparseTerms(review_dtm_neg, sparsity)


######################################################################################################################## 
################################################### TFIDF ############################################################## 
######################################################################################################################## 

######################################### INCREASE IMPORTANCE OF POSITIVE WORDS ######################
# Objective: To increase the importance of positive words by the frequency it appears in negative documents
# The lower the number of times it appears in negative documents, the weight of the positive word will be increased
# Term Frequency (Normalized) - First part of equation: Divide frequency of words in positive documents with the length of the positive document the word is in
pos_tf = as.matrix(review_dtm_pos) # Convert to matrix with each term in a column
pos_tf = data.frame(pos_tf)
pos_tf = pos_tf/rowSums(pos_tf, na.rm = T) # Normalize
pos_tf[is.na(pos_tf)] = 0

# IDF - Second part of equation (Count N and dw) 
neg_idf = as.matrix(review_dtm_neg) # Convert to matrix with each term in a column
neg_idf = slam::col_sums(neg_idf/neg_idf, na.rm = T) # dw - Count the number of negative documents containing the word
neg_idf = data.frame(neg_idf) # Convert to dataframe
neg_idf$term = rownames(neg_idf) # Add column containing the word
neg_idf$secondlog = 1 + log1p(nrow(review_dtm_neg)/neg_idf$neg_idf) # ln

unique_poswords = data.frame(colnames(review_dtm_pos))
colnames(unique_poswords) = c("term")

pos_idftable = left_join(unique_poswords,neg_idf, by = c("term"))

# As the second log for positive terms will become infinity when it has not appeared in negative documents at all,
# below formula is used to extrapolate the value of not appearing at all [1+log(3000/0)] by using the below two values
# [1+log(3000/1)] and [1+log(3000/2)] 
y1 =1+log(nrow(review_dtm_neg)/1)
y2= 1+log(nrow(review_dtm_neg)/2)
grad = (y1-y2)/(1-2)
y0 = y1 + grad*-1

pos_idftable$secondlog[is.na(pos_idftable$secondlog)] = y0 # replace NA values with extrapolated value for those positive words that have not appeared in negative documents at all
pos_idftable = as.data.frame(matrix(pos_idftable$secondlog)) # keep only value of second log

pos_tf = sapply(pos_tf[, ], as.numeric)
pos_tfidf = pos_tf*pos_idftable[col(pos_tf)] # Compute TFIDF - Multiply TF with IDF

# wordcloud
wordcloud(colnames(pos_tfidf), colSums(pos_tfidf[,-1]), max.words=50, colors=brewer.pal(3, "Dark2"))


######################################### INCREASE IMPORTANCE OF NEGATIVE WORDS ######################
# Objective: To increase the importance of negative words by the frequency it appears in positive documents
# The lower the number of times it appears in positive documents, the weight of the negative word will be increased
# Term Frequency (Normalized) - First part of equation: Divide frequency of words in negative documents with the length of the negative document the word is in
neg_tf = as.matrix(review_dtm_neg) # Convert to matrix with each term in a column
neg_tf = data.frame(neg_tf)
neg_tf = neg_tf/rowSums(neg_tf, na.rm = T) # Normalize
neg_tf[is.na(neg_tf)] = 0

# IDF - Second part of equation (Count N and dw) 
pos_idf = as.matrix(review_dtm_neg) # Convert to matrix with each term in a column
pos_idf = slam::col_sums(pos_idf/pos_idf, na.rm = T) # dw - Count the number of positive documents containing the word
pos_idf = data.frame(pos_idf) # Convert to dataframe
pos_idf$term = rownames(pos_idf) # Add column containing the word
pos_idf$secondlog = 1 + log1p(nrow(review_dtm_pos)/pos_idf$pos_idf) # ln

unique_negwords = data.frame(colnames(review_dtm_neg))
colnames(unique_negwords) = c("term")

neg_idftable = left_join(unique_negwords,neg_idf, by = c("term"))

neg_idftable$secondlog[is.na(neg_idftable$secondlog)] = y0 # replace NA values with extrapolated value for those positive words that have not appeared in negative documents at all
neg_idftable = as.data.frame(matrix(neg_idftable$secondlog)) # keep only value of second log

neg_tf = sapply(neg_tf[,], as.numeric)
neg_tfidf = neg_tf*neg_idftable[col(neg_tf)] # Compute TFIDF - Multiply TF with IDF


wordcloud(colnames(neg_tfidf), colSums(neg_tfidf[,-1]), max.words=50, colors=brewer.pal(3, "Dark2"))


######################################################################################################################## 
################################################### DATASET PREPARATION FOR MODEL BUILDING ############################# 
######################################################################################################################## 

# Combine sentiment with matrix
Pos_df = as.data.frame(pos_tfidf) %>% mutate(Sentiment = 1) 
Neg_df = as.data.frame(neg_tfidf) %>% mutate(Sentiment = 0) 
NegPos_df = rbind.fill(Pos_df,Neg_df) # rbind both positive and negative tfidf
NegPos_df[is.na(NegPos_df)] = 0 # if na, fill with 0
NegPos_df$Sentiment = as.factor(NegPos_df$Sentiment) 

# Create two additional terms using positive and negative words dictionary
NegPos_df$pos = tm_term_score(review_dtm_pos, pos_words_bi)
NegPos_df$neg = tm_term_score(review_dtm_neg, neg_words_bi)

#   Split data into 70% training 30% test
train_ind = sample(nrow(NegPos_df),nrow(NegPos_df)*0.70)
test = NegPos_df[-train_ind,]
train = NegPos_df[train_ind,]


######################################################################################################################## 
################################################### MODELING ########################################################### 
######################################################################################################################## 

######################################### MODELING - LOGISTIC REGRESSION #############################
# Train model
reviews_lr = glm(Sentiment~ ., family = binomial, data = train, maxit = 100)

# when looking at summary of m_logit, the coefficient section lists all the input variables used in the model. A series of asterisks at the very end of them gives us the importance of each one, with *** being the greatest significance level, and ** or * being also important. These starts relate to the values in Pr
summary(reviews_lr)

# Apply model to test dataset to evaluate performance
pred_lr = as.numeric(predict(reviews_lr, test, type="response") > 0.5)

######################################### MODELING - NAIVE BAYES #####################################
# Train Model
reviews_nbayes = naiveBayes(Sentiment ~ ., data=train, threshold=.5)

# Apply model to test dataset to evaluate performance
pred_nbayes = predict(reviews_nbayes, test, threshold=.5)

####################################### MODELING - SUPPORT VECTOR MACHINE ###########################
#Train Model
reviews_svm = svm(Sentiment~., data = train)

# Apply model to test dataset to evaluate performance
pred_svm = predict(reviews_svm,test)

####################################### MODELING - NEURAL NETWORK ###################################
#Train Model
reviews_nnet = nnet(Sentiment~., data=train, size=1, maxit=500)

# Apply model to test dataset to evaluate performance
prob_nnet= predict(reviews_nnet,test)
pred_nnet = as.numeric(prob_nnet > 0.5)

######################################### MODELING - DECISION TREE ##################################
#Train Model
reviews_tree = rpart(Sentiment~.,  method = "class", data = train);  
prp(reviews_tree)

# Apply model to test dataset to evaluate performance
pred_tree = predict(reviews_tree, test,  type="class")


######################################################################################################################## 
################################################### EVALUATION ######################################################### 
################################################### Accuracy, True Negative, Precision, True Positive, False Positive ##

# Create function
sensitivity = function(confusion_matrix){
  acc = (confusion_matrix[1]+confusion_matrix[4]) / sum(confusion_matrix)
  tn = confusion_matrix[1] / (confusion_matrix[3]+confusion_matrix[1])
  ppv = confusion_matrix[4] / (confusion_matrix[4]+confusion_matrix[3])
  tp = confusion_matrix[4] / (confusion_matrix[4]+confusion_matrix[2])
  fpr = confusion_matrix[3] / (confusion_matrix[3]+confusion_matrix[1])
  return(list(accuracy=acc, specificity=tn, precision=ppv, sensitivity=tp, fpr=fpr))
}

# Evaluate Performance
cm_lr = sensitivity(table(test$Sentiment,pred_lr,dnn=c("Obs","Pred"))) # Logistic Regression
cm_svm = sensitivity(table(test$Sentiment,pred_svm,dnn=c("Obs","Pred"))) # Support Vector Machine
cm_nbayes = sensitivity(table(test$Sentiment, pred_nbayes, dnn=c("Obs","Pred"))) # Naive Bayes
cm_nnet = sensitivity(table(test$Sentiment, pred_nnet, dnn=c("Obs","Pred"))) # Neural Network
cm_dtree = sensitivity(table(test$Sentiment,pred_tree,dnn=c("Obs","Pred"))) # Decision Tree


