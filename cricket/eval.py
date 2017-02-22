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

os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data')









sumtext = ['/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match1_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match2_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match3_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match4_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match5_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match6_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match7_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match8_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match9_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match10_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match11_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match12_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match13_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match14_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match15_summary.txt',
            '/home/anirudh/projects/BTP_Text_Summarization/code/data/TextRankSummary/match16_summary.txt']
reftext = [['/home/anirudh/projects/BTP_Text_Summarization/code/data/match1_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match2_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match3_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match4_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match5_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match6_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match7_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match8_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match9_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match10_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match11_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match12_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match13_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match14_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match15_summary.txt'],
            ['/home/anirudh/projects/BTP_Text_Summarization/code/data/match16_summary.txt']]

recall, precision, F_measure = PythonROUGE.PythonROUGE(sumtext, reftext, ngram_order=2)
print recall, precision, F_measure
