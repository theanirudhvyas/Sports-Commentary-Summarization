# The cricket commentary scraped from http://www.cricbuzz.com and was acquired in the form of a text file. This code segments the sentences of the text file, generates
#specific features for each sentence and finally writes a csv file.


import nltk
import os
import sys
import re
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#appending path for pythonRouge package. The package computes ngram similarity
#between sentences and the similarity thus acquired is used as target variable and evaluation metric for the model.
sys.path.append(os.path.abspath("/home/anirudh/projects/BTP_Text_Summarization/code/PythonROUGE"))
import PythonROUGE

os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data')
for filenum in range(10, 17):
    print filenum
    filename = 'match' + str(filenum) + '.txt'
    comm = open(filename, 'r')

    text = comm.read()
    text = re.sub(r'\d+.\d', ' . ', text)
# tokenize sentences from the text.
    sent = sent_tokenize(text)
    sent = [s.strip() for s in sent]

    df = pd.DataFrame()
    df['sentences'] = sent

# the number of sentences in a single commentary. This is used to compute the position of a sentence in the document.
    num = len(sent)
# the position of a sentence in  the document. Sentences that occur at the start or end of the document are more likely to come into the news summary
    pos = [num-(i+1) for i in range(num)]
    df['pos'] = pos
#initializing english stopwords
    stop = set(stopwords.words('english'))

    length = [0] * num
    len_stop = [0]*num

    tf = [0]*num

#calculating tfidf values of words in each sentence.
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(sent)
    idf = vectorizer._tfidf.idf_
    tfidf = dict(zip(vectorizer.get_feature_names(), idf))


    for i in range(num):
        token = word_tokenize(df.loc[i]['sentences'])
        token_stop = [w for w in token if w.lower() not in stop]
        length[i] = len(token_stop)
        len_stop[i] = len(token) - len(token_stop)
        tf[i] = 0
        for w in token:
            if w.lower() in tfidf.keys():
                tf[i] += tfidf[w.lower()]

    df['len'] = length
    df['len_stop'] = len_stop
    df['tfidf'] = tf

    sim = (X*X.T).A
#
    sim_p = [0]*num
    sim_pp = [0]*num
    sim_n = [0]*num
    sim_nn = [0]*num
#calculating similarity of sentences with previous 2 and next 2 sentences
    sim_p[1] = sim[1][0]
    for i in range(2, num):
        sim_p[i] = sim[i][i-1]
        sim_pp[i] = sim[i][i-2]

    sim_n[num-2] = sim[num-2][num-1]
    for i in range(num-2):
        sim_n[i] = sim[i][i+1]
        sim_nn[i] = sim[i][i+2]

    df['sim_p'] = sim_p
    df['sim_pp'] = sim_pp
    df['sim_n'] = sim_n
    df['sim_nn'] = sim_nn



# checking for some domain specific words and their occurence in each sentence
    df['century'] = [1 if 'century' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['out'] = [1 if 'out' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['bold'] = [1 if 'bold' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['caught'] = [1 if 'caught' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['stumped'] = [1 if 'stumped' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['wicket'] = [1 if 'wicket' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['hattrick'] = [1 if 'hattrick' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['catch'] = [1 if 'catch' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['six'] = [1 if 'six' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['four'] = [1 if 'four' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['amazing'] = [1 if 'amazing' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['hundred'] = [1 if 'hundred' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['ton'] = [1 if 'ton' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['boundary'] = [1 if 'boundary' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['drop'] = [1 if 'drop' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['toss'] = [1 if 'toss' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['won'] = [1 if 'won' in x.lower() else 0 for x in df.iloc[:]['sentences']]
    df['lost'] = [1 if 'lost' in x.lower() else 0 for x in df.iloc[:]['sentences']]

    # create target variable
    filename = 'match' + str(filenum) + '_summary.txt'
    su = open(filename, 'r')
    summ = sent_tokenize(su.read())
    score = [0]*len(sent)
    temp = [0]*len(summ)
    sumtext = ['/home/anirudh/projects/BTP_Text_Summarization/code/data/summary.txt']
    reftext = [['/home/anirudh/projects/BTP_Text_Summarization/code/data/reference.txt']]
    for i, sen in enumerate(df.loc[:]['sentences']):
        with open('reference.txt', 'r+') as f:
            text = f.read()
            text = sen
            f.seek(0)
            f.write(text)
            f.truncate()
        for j, ref in enumerate(summ):
            with open('summary.txt', 'r+') as f:
                text = f.read()
                text = ref
                f.seek(0)
                f.write(text)
                f.truncate()

            recall, precision, F_measure = PythonROUGE.PythonROUGE(sumtext, reftext, ngram_order=2)
            temp[j] = F_measure
        avg = [(d[0] + d[1])/2 for d in temp]
        score[i] = max(avg)

    df['score'] = score
    filename = 'match_' + str(filenum) + '.csv'
    df.to_csv(filename)
