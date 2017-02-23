# Sports-Commentary-Summarization
Creating news articles from cricket and football commentaries.

1. OVERVIEW
The objective of this project is to generate summarized news articles from cricket match commentaries and match statistics. Since,  patterns can be observed in human written summaries or news articles, the task of auto-summarization can be achieved using Natural language processing and Machine learning techniques.
I am attempting extractive Summarization, as generating natural language is a herculean task in itself. 
Input
We take the live commentary of a sport(we focus on cricket here), in a script(text format) to generate the news. For analysis, development and evaluation of the model we are also taking the news documents associated to their corresponding commentaries.


Output
After processing the input, we give a subset of the sentences of the commentary as a summarization of the commentary. The sentences are selected such that they strive to cover all the information relevant to the match and resemble the news article for that match.

1. APPROACH
I am using a supervised learning algorithm here, inspired from [1] . In unsupervised summarization algorithms like text rank and lex rank features that are context-specific and related to domain knowledge of the sport are not considered for building a model for sports summarization. Since human written summaries are readily available in the form of news article for sports, we can use them as training target vectors and thus improve the quality of automatically generated summaries. Hence, by training a supervised learning model, better results can be achieved as compared to rule based or unsupervised learning.

Features Extraction
Following features were extracted from the cricket match summary data, based on
Length of the sentence: Too short sentences are not included in the summary
Position of sentence: Sentences that are at the end of each innings have more probability of being in the summary. As the commentator summarizes the outstanding events in the innings.
Length after stopwords Removal: Stop words are non-contextual words like ‘a’, ‘and’,’the’  and hence are not important in summarizing the meaning.
Cosine Similarity to Previous sentence, Previous to previous sentence, next sentence and next to sentence:  Coherent and informed summaries are provided suing these features.
 Count of buzz words: Buzz words like “century”, ”hat-trick”, “bowled”, ”won”, ”loss”, ”wicket”, ”six”, ”innings”, “score”, “target” are frequently occurring words in the summary. These words impart domain knowledge to the training model.


Target Variable : To get the target variable we took the maximum (rouge) similarity of every sentence in the corpus with each sentence of the corresponding news. The target variable lies between 0 and 1. This is a good choice for the target variable as described in [1]
Training Model:
Training was performed using Random Forest regression model. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
500 decision tree random forest was used for training and R’s randomForest package was utilized. Error rate graph is shown below:


Cricket VS Football

The data obtained from crickbuzz was very noisy and the commentary was very repetitive and so not useful for our project.
Thus the results obtained were not satisfactory and so i decided to crosscheck the algorithm with football data.

The results were similar and thus we can conclude that the algorithm is not capable enough to generate human readable summaries.


CURRENT WORK

So now I am studying and trying to implement knowledge graphs and information extraction methods to generate news.



REFERENCES
Zhang, Jianmin, Jin-ge Yao and Xiaojun Wan. “Toward Constructing Sports News from Live Text Commentary”, Proceedings of ACL, 2016
