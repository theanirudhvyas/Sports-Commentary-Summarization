import nltk
import os
import sys

import re
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append(os.path.abspath("/home/anirudh/projects/BTP_Text_Summarization/code/PythonROUGE"))
import PythonROUGE

os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football')




summary = '/home/anirudh/projects/BTP_Text_Summarization/code/data/Football_summary/football_summary_%d.txt'




sumtext = [[summary%(1)],
            [summary%(3)],
            [summary%(4)],
            [summary%(5)],
            [summary%(6)],
            [summary%(7)],
            [summary%(8)],
            [summary%(9)],
            [summary%(10)],
            [summary%(11)],
            [summary%(12)],
            [summary%(13)],
            [summary%(14)],
            [summary%(15)],
            [summary%(16)],
            [summary%(17)],
            [summary%(18)],
            [summary%(19)],
            [summary%(20)],
            [summary%(21)],
            [summary%(22)],
            [summary%(23)],
            [summary%(24)],
            [summary%(25)],
            [summary%(26)],
            [summary%(27)],
            [summary%(28)],
            [summary%(29)],
            [summary%(30)],

            ]

reference = '/home/anirudh/projects/BTP_Text_Summarization/code/data/data_football/match_%d_report.txt'

reftext = [[reference%(1)],
            [reference%(1)],
            [reference%(3)],
            [reference%(4)],
            [reference%(5)],
            [reference%(6)],
            [reference%(7)],
            [reference%(8)],
            [reference%(9)],
            [reference%(10)],
            [reference%(11)],
            [reference%(12)],
            [reference%(13)],
            [reference%(14)],
            [reference%(15)],
            [reference%(16)],
            [reference%(17)],
            [reference%(18)],
            [reference%(19)],
            [reference%(20)],
            [reference%(21)],
            [reference%(22)],
            [reference%(23)],
            [reference%(24)],
            [reference%(25)],
            [reference%(26)],
            [reference%(27)],
            [reference%(28)],
            [reference%(29)],
            [reference%(30)],
                        ]

recall, precision, F_measure = PythonROUGE.PythonROUGE(sumtext, reftext, ngram_order=2)
print recall, precision, F_measure
