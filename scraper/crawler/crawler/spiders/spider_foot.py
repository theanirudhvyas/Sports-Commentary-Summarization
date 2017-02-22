#A spider to scrape through the sportsmole website to get the commentary of a football game in csv format and its corresponding news document

import scrapy
from bs4 import BeautifulSoup
import pandas as pd
import re

class qSpider(scrapy.Spider):
	name = "commentary"

	def start_requests(self):
		urls = ['http://www.sportsmole.co.uk/football/liverpool/live-commentary/live-commentary-sunderland-vs-liverpool_288522.html']
		for url in urls:
			yield scrapy.Request(url = url, callback = self.parse)


	def parse(self, response):
		ls = []
		comm = BeautifulSoup(response.text, 'lxml')
		for comment in comm.find_all('div', {'class':'livecomm'}):
			new = [ comment.a.text, comment.span.text]	
			ls.append(new)

		labels = ['time', 'text']
		df = pd.DataFrame.from_records(ls, columns=labels)
		df.to_csv('match_31_comm.csv', encoding='utf-8')

		#report = response.css('ul.game_links a::attr(href)').extract()
		a = comm('a', text = re.compile('match report', re.IGNORECASE))
		if a:
			link = a[0].get('href')
			rep = response.urljoin(link)
			yield scrapy.Request(rep, callback=self.read_report)

	def read_report(self, response):
			website = BeautifulSoup(response.text, 'lxml')
			title = website.find(id='title_text').string
			text = title + '\n'
			for p in website.find_all('p'):
				text = text + " " + p.text
			filename = 'match_31_report.txt'
			with open(filename, 'wb') as f:
				f.write(text.encode('utf-8'))
			return(None)





