# In the football data acquired from the spider, each row of the dataset had multiple sentences. This code fixes that problem and segments the sentences 
#and attributes a time-stamp for each sentence
import pandas as pd
import numpy as np
import os
from nltk import sent_tokenize, word_tokenize

#changing the working directory to the data dir
os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football')

for i in range(1,31):
    filename = "match_" + str(i) + "_comm.csv"
    print filename
    df = pd.read_csv(filename)

    df = df.drop('Unnamed: 0', axis=1) #deleting the index row, which is useless
    new = pd.DataFrame(columns = ['time', 'text'])


    #for iterrating through all the rows of the DataFrame
    for index, row in df.iterrows():
        if row['text'] is not np.NAN and len(row['text'].strip()):
            sent = [s.strip().encode('utf-8') for s in sent_tokenize(row['text'].decode('utf-8'))]
            d = {'time' : list([row['time']]*len(sent)), 'text' : sent}
            temp = pd.DataFrame(d)
            new = new.append(temp)

    new.to_csv("./cleaned_data/clean_" + filename)
