# Sports-Commentary-Summarization
Creating news articles from cricket and football commentaries.


SPORTS COMMENTARY

SUMMARIZATION

1. OVERVIEW

The objective of this project is to generate summarized news articles from cricket match

commentaries and match statistics. Since, patterns can be observed in human written

summaries or news articles, the task of auto-summarization can be achieved using Natural

language processing and Machine learning techniques.

1.1 Extraction-based summarization

In this summarization task, the algorithm extracts objects from the entire collection, without

modifying the objects themselves. Examples of this include keyphrase extraction, where the

goal is to select individual words or phrases to "tag" a document, and document

summarization, where the goal is to select whole sentences (without modifying them) to create

a short paragraph summary. Similarly, in image collection summarization, the system extracts

images from the collection without modifying the images themselves.

2. Problem Statement

Can we automatically generate news articles from cricket matches’ live commentaries (text

scripts) such that they resemble the news written by professional sports journalists and give as

much information about the match as possible in a predefined number of sentences?

3. Input and Output

● Input

○ We take the live commentary of a sport(we focus on cricket here), in a script(text

format) to generate the news. For analysis, development and evaluation of the

model we are also taking the news documents associated to their corresponding

commentaries.

● Output

○ After processing the input, we give a subset of the sentences of the commentary

as a summarization of the commentary. The sentences are selected such that

they strive to cover all the information relevant to the match and resemble the

news article for that match.

4. APPROACH

The average number of sentences in a single match is 700, while the number of sentences in a

news article are 33. Given the stark difference in the number of sentences and the large size of

the corpus, extractive summarization is an apt approach for the problem. Abstractive

summarization fails to give meaningful results for texts larger than a few hundred words (Peter

Liu and Xin Pan, 2016) [9].

At a very superficial level, summarization algorithms try to find subsets of objects (like set of

sentences, or a set of images), which cover information of the entire set. This is also called the

core-set. Some techniques and algorithms which naturally model summarization problems are

TextRank and PageRank, Submodular set function, Determinantal point process [7], maximal

marginal relevance (MMR) etc.

Therefore, formally defining the problem statement,

Given a piece of live text commentary containing a collection of candidate sentences S = {s1,

s2, . . . , sn} describing a particular sports game G, we will extract sentences to form a summary

of G which will be suitable to be formed as sports news. The total length will not exceed a

pre-specified length B.

4.1 Unsupervised learning approaches

4.1.1 Text Rank

Following are the steps of Text Rank algorithm for text summarization.

1. Tokenize the text into sentences

2. Tokenize each sentence into a collection of words

3. Convert the sentences into graphs

A graph is constructed by creating a vertex for each sentence in the document. , The

edges between sentences are based on content overlap.TextRank uses a very similar

measure based on the number of words two sentences have in common [4]

4. Score the sentences via pagerank

Page-rank formulation PR(pi

) is the page-rank score assigned to page pi

(8)

where is the total number of nodes in the graph, and is a ``damping factor", which is

typically chosen in the interval [0.1 0.2]

5. Extract the given number of sentences with maximum page-rank scores

4.1.2 Lex Rank

LexRank [4] is an algorithm essentially identical to TextRank, and both use this approach for

document summarization.

1. Tokenize the text into sentences

2. Tokenize each sentence into a collection of words

3. Convert the sentences into graphs.

A graph is constructed by creating a vertex for each sentence in the document.

The edges between sentences are based on Cosine similarity of word vectors which are then

converted to TF-IDF scores which are row-wise normalised.

We can normalize the row sums of the corresponding transition matrix so that we have a

stochastic matrix. The resultant equation is a modified version of LexRank for weighted graphs:

(10)

This way, while computing LexRank for a sentence, we multiply the LexRank values of the

linking sentences by the weights of the links. Weights are normalized by the row sums, and the

damping factor is added for the convergence of the method.

Cosine Similarity ​of two sentence vectors A ​and B

Where Ai

and Bi

are components or TF-IDF​ scores of words in sentences A ​and B​ respectively.

4.2 Rule Based Approach

4.2.1 Motivation

The website (www.espncricinfo.com) has a lot more information than text scripts (summaries). It

gives summaries and statistics of every over, innings statistics and individual player

performances. The information is provided in a standard template format and can be utilized by

scraping the website for specific tags

4.2.2 Algorithm

1. We first find the tag that gives the final result of the match, and the standings in the

series (if available)

2. Then we find the tag corresponding to the toss result, and who played the first innings

3. Then we find scores of the teams at 5th, 10th and 15th overs of their respective innings

(we are considering a 20-20 format cricket match here)

4. Finally we use part of speech tagging to identify the names of the players from the text

and add specific player statistics to the summary

4.2.3 Pros and Cons

● Pros

○ The algorithm uses processed information rather than raw summaries and thus

gives a more accurate summary

○ The summaries produced and concise and lexically coherent.

○ The length of the summary is predefined and information provided is complete.

● Cons

○ The algorithm proposed is not robust, in the sense that even if one of the tags

used by the website is changed or removed the summaries would be

incomplete

○ The algorithm is very specific to source of information, i.e. we can only take data

from a specific website (www.espncricinfo.com) and a specific format (20-20) of

a single sport (cricket).

○ The algorithm cannot be scaled for other sports or other formats without

significantly changing it.

4.2 Supervised Learning Approach

4.2.1 Motivation

In Unsupervised learning i.e Text Rank and Lex Rank algorithms generic features and text

similarity is taken into account. Features that are context-specific and related to domain

knowledge of the sport must also be considered for building a model for sports summarization.

Since human written summaries are readily available in the form of news article for sports, we

can use them as training target vectors and thus improve the quality of automatically generated

summaries. Hence, by training a supervised learning model, better results can be achieved as

compared to rule based or unsupervised learning.

4.2.2 Data Collection

1. Cricket match commentary data from 30 matches of ICC World T20 series was

​ collected from www.cricbuzz.com by using web scraping techniques. Python packages-
scrapy and beautifulSoup were used for web scraping.

2. Cricket match news articles or the reference summaries were also scraped from

www.cricbuzz.com using the above mentioned packages.

4.2.3 Data description:

1. Out of 30, 20 matches​ were used for training data an ​ d remaining 10 matches were

used for final testing o​ f the model.

2. Each match had on average 700 sentences and each sentence served as a sample

data point for training data set. Thus around 14,000 data points were generated.

4.2.4 Features Extraction

Following features were extracted from the cricket match summary data, based on [5]

​ 1. Length of the sentence: Too short sentences are not included in the summary

​ 2. Position of sentence: Sentences that are at the end of each innings have more

probability of being in the summary. As the commentator summarizes the outstanding

events in the innings.

3. Length after stopwords Removal​: Stop words are non-contextual words like ‘a’,

‘and’,’the’ and hence are not important in summarizing the meaning.

4. Cosine Similarity​ to Previous sentence, Previous to previous sentence, next sentence

and next to sentence: Coherent and informed summaries are provided suing these

features.

5. Count of buzz words: ​Buzz words like “century”, ”hat-trick”, “bowled”, ”won”, ”loss”,

”wicket”, ”six”, ”innings”, “score”, “target” are frequently occurring words in the

summary. These words impart domain knowledge to the training model.

6. Target Variable : ​To get the target variable we took the maximum (rouge) similarity of

every sentence in the corpus with each sentence of the corresponding news. The

target variable lies between 0 and 1. This is a good choice for the target variable as

described in [1]

4.2.5 Training Model:

Training was performed using Random Forest​ [8] regression ​model. Random forests or

random decision forests are an ensemble learning method for classification, regression and

other tasks, that operate by constructing a multitude of decision trees at training time and

outputting the class that is the mode of the classes (classification) or mean prediction

(regression) of the individual trees. Random decision forests correct for decision trees' habit of

overfitting to their training set.

500 decision tree​ random forest was used for training and R’s randomForest​ package was

utilized. Error rate graph is shown below:

4.2.6 Validation:

​3 fold cross validation was applied on the 20 training match data.

4.2.7 Summary Generation:

Since average number of sentences in the reference summary was 30. 30 sentences having

highest ROUGE [3] score (closer to 1) values after training the model were used as extractive

summary sentences.

5. Evaluation

Before getting into the details of some summarization methods, we will mention how

summarization systems are typically evaluated. The most common way is using the so-called

ROUGE​ (Recall-Oriented Understudy for Gisting Evaluation)​ measure [3]. This is a

recall-based measure that determines how well a system-generated summary covers the

content present in one or more human-generated model summaries known as references. It is

recall-based to encourage systems to include all the important topics in the text. Recall can be

computed with respect to unigram, bigram, trigram, or 4-gram matching. For example,

ROUGE-1 is computed as division of count of unigrams in reference that appear in system and

count of unigrams in reference summary.

6. MILESTONES

6.1 Background Research on summarization Techniques

Radev et al. (2004) pioneered the use of cluster centroids in their work with the idea to group,

in the same cluster, those sentences which are highly similar to each other, thus generating a

number of clusters. To measure the similarity between a pair of sentences, the authors use the

cosine similarity measure where sentences are represented as weighted vectors of tf-idf terms.

Once sentences are clustered, sentence selection is performed by selecting a subset of

sentences from each cluster. In TextRank [4] ​(2004), a document is represented as a graph

where each sentence is denoted by a vertex and pairwise similarities between sentences are

represented by edges with a weight corresponding to the similarity between the sentences.

The Google PageRank[4]​ ranking algorithm is used to estimate the importance of different

sentences and the most important sentences are chosen for inclusion in the summary.

6.2 Implementation of Basic Model

Implemented basic summarization techniques , word frequency and lexrank to be precise.

Evaluated the models and concluded that LexRank gave better results in all the scenarios that

we tested.

6.3 Research for Sport Commentary summarization

Studied the research paper Towards Constructing Sports News from Live Text Commentary -

​ Jianmin Zhang, Jin-ge Yao, et al [1] . They built a model for summarizing football

commentaries in Chinese. They proposed a supervised learning approach with rogue-2F

scores as the target variable. Several Features, both generic and domain specific, are extracted

and used for training. They also used a probabilistic approach for sentence extraction to avoid

redundancy in the summary.

A subjective summarization approach was also studied from the paper: Two sides to every

story: Subjective event summarization of sports events using Twitter - David Corney, et al.

​ [2] They proposed a summarization technique based on tweets by users and commentary.

They classify the users as supporters of a particular team and then under the assumption that a

user gives correct and subjective information about a particular match, summarize the

information.

6.4 Building a Rule based model

Built a programme that scrapes espn website to access commentary and match statistics and

then proceeds to summarize the match based on rules and natural language processing

models. The programme uses POS (part of speech tagging)[6] ​to identify player names. The

programme acts as a good measure of the breadth and depth of the problem and gives an

acceptable result even though it is restricted to a particular game and source of information

6.5 Building an Unsupervised Learning model (Lex Rank and Text Rank):

Methods to find importance or prestige of each sentence in the training commentary dataset

and select the sentences with highest prestige values. These importance are found by Graph

based centrality measures in case of Text Rank - Degree centrality (depends on the degrees

of adjacent sentences) and in case of Lex Rank - Cosine Similarity with the adjacent sentences.

Text Rank [10]​ ranking is done on the basis of its number of connections (degree of node) with

neighbouring sentences (sentences with common lexical items).

LexRank[4]​, for computing sentence importance based on the concept of eigenvector

centrality in a graph representation of sentences. In this model, a connectivity matrix based on

intra-sentence cosine similarity is used as the adjacency matrix of the graph representation of

sentences.

6.6 Building a Supervised Learning model:

Based on the merits and shortcomings of the above models we decided to model the problem

as a supervised learning problem, for that we extracted specific generic and domain specific

features (as described in [1]) and a target variable (as described in 4.2.4). We trained and tested

the model and the results are mention in section 7.

7. Results

Comparison of all models used for summarization:

In all these models, the ROUGE -1 score is used as a measure for relevance of

sentence with the reference summary. Thus it is found that Lex Rank performs

better than Supervised model, due to the fact that better domain specific features

can be evaluated and proper pre-processing can be done using stemming and

parsing.

8. FUTURE WORK

1. We plan to improve the supervised learning method by incorporating a better

dataset and extracting more domain specific features.

2. Improve on the research mentioned in the paper, by evaluating different

algorithms for training and evaluation - using stemming an ​ d vocabulary

(WordNet) for getting similar words.

3. We plan to apply a probabilistic model for sentence selection (Detrimental

Point Processes​) [1][7] to avoid global redundancy.

4. Implementing continuous feature space representation using Word2Vec

technique for text summarization.

5. Finally our main motive is to generalize the model to accept multiple sports.

9. REFERENCES

1. Zhang, Jianmin, Jin-ge Yao and Xiaojun Wan. “Toward Constructing Sports News from Live Text

Commentary”, Proceedings of ACL, 2016

2. Corney, D., Martin, C. and Göker, A., 2014, April. Two Sides to Every Story: Subjective Event

Summarization of Sports Events using Twitter. In SoMuS@ ICMR.

3. Lin, C.Y., 2004, July. Rouge: A package for automatic evaluation of summaries. In Text

summarization branches out: Proceedings of the ACL-04 workshop (Vol. 8).

4. Erkan, G. and Radev, D.R., 2011. Lexrank: Graph-based lexical centrality as salience in text

summarization. CoRR, abs/1109.2128.

5. Gambhir, M. and Gupta, V., 2016. Recent automatic text summarization techniques: a survey.

Artificial Intelligence Review, pp.1-66.

6. Toutanova, K. and Manning, C.D., 2000, October. Enriching the knowledge sources used in a

maximum entropy part-of-speech tagger. In Proceedings of the 2000 Joint SIGDAT conference on

Empirical methods in natural language processing and very large corpora: held in conjunction

with the 38th Annual Meeting of the Association for Computational Linguistics-Volume 13 (pp.

63-70). Association for Computational Linguistics.

7. Kulesza, A. and Taskar, B., 2012. Determinantal point processes for machine learning. arXiv

preprint arXiv:1207.6083.

8. A. Liaw and M. Wiener (2002). Classification and Regression by randomForest. R News 2(3),

18--22.

9. https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html

10. Mihalcea, R. and Tarau, P., 2004, July. TextRank: Bringing order into texts. Association for

Computational Linguistics.
