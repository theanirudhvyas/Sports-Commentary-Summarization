# Generating summaries with well know alorithms like textRank and LexRank.
import os
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in

os.chdir('/home/anirudh/projects/BTP_Text_Summarization/code/data')
for number in range(1,17):
    file = 'match' + str(number) + '.txt' #name of the plain-text file
    parser = PlaintextParser.from_file(file, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, 30) #Summarize the document with 30 sentences

    target = 'summary_' + str(number) + '.txt'
    with open(target, 'w') as f:
        for sentence in summary:
            f.write(str(sentence))
