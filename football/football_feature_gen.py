# the football data acquired from the spider is in csv format. In this code we generate features for each sentence similar to those
#generated in cricket.
import nltk
import os
import sys
import re
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#importing PythonRouge package. We use the package to calculate the target variable and also as evaluation metric for the model
sys.path.append(os.path.abspath("/home/anirudh/projects/BTP_Text_Summarization/code/PythonROUGE"))
import PythonROUGE

os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football/cleaned_data')

for filenum in range(14, 31):

    filename = "clean_match_"+ str(filenum) + "_comm.csv"
    print filename
    df = pd.read_csv(filename)

    df = df.drop('Unnamed: 0', axis=1) #deleting the index row, which is useless

    num = len(df)

    stop = set(stopwords.words('english'))

    length = [0] * num
    len_stop = [0]*num
    tf = [0]*num

    vectorizer = TfidfVectorizer(min_df=1)

    X = vectorizer.fit_transform(df['text'])
    idf = vectorizer._tfidf.idf_
    tfidf = dict(zip(vectorizer.get_feature_names(), idf))

    for i in range(num):
        token = word_tokenize(df.loc[i]['text'].decode('utf-8'))
        token_stop = [w for w in token if w.lower() not in stop]
        length[i] = len(token_stop)
        len_stop[i] = len(token) - len(token_stop)
        tf[i] = 0
        for w in token:
            if w.encode('utf-8').lower() in tfidf.keys():
                tf[i] += tfidf[w.decode('utf-8').lower()]


    df['len'] = length
    df['len_stop'] = len_stop
    df['tfidf'] = tf

    sim = (X*X.T).A

    sim_p = [0]*num
    sim_pp = [0]*num
    sim_n = [0]*num
    sim_nn = [0]*num


    sim_p[1] = sim[1][0]
    for i in range(2,num):
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

    df['goal'] = [1 if 'goal' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['score'] = [1 if 'scored' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['won'] = [1 if 'won' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['win'] = [1 if 'win' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['lose'] = [1 if 'lose' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['lost'] = [1 if 'lost' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['draw'] = [1 if 'draw' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['penalty'] = [1 if 'penalty' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['goal'] = [1 if 'goal' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['position'] = [1 if 'position' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['breakthrough'] = [1 if 'breakthrough' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['header'] = [1 if 'header' in x.lower() else 0 for x in df.iloc[:]['text']]
    df['offside'] = [1 if 'offside' in x.lower() else 0 for x in df.iloc[:]['text']]


    # create target variable
    file_report = "/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football/match_"+ str(filenum) + "_report.txt"
    print file_report
    su = open(file_report, 'r')
    summ = sent_tokenize(su.read().decode('utf-8'))
    score = [0]*num
    temp = [0]*len(summ)
    sumtext = ['/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football/cleaned_data/summary.txt']
    reftext = [['/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football/cleaned_data/reference.txt']]
    for i,sen in enumerate(df.loc[:]['text']):
        with open('reference.txt', 'r+') as f:
            text = f.read()
            text = sen
            f.seek(0)
            f.write(text.decode('utf-8').encode('utf-8'))
            f.truncate()
        for j,ref in enumerate(summ):
            with open('summary.txt', 'r+') as f:
                text = f.read()
                text = ref
                f.seek(0)
                f.write(text.encode('utf-8'))
                f.truncate()

            recall,precision,F_measure = PythonROUGE.PythonROUGE(sumtext, reftext, ngram_order=2)
            temp[j] = F_measure
        avg = [(d[0] + d[1])/2 for d in temp]
        score[i] =  max(avg)

    df['score'] = score
    out_file = "match_" + str(filenum) + ".csv"
    print out_file
    df.to_csv(out_file)
