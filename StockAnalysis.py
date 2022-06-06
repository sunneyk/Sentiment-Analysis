from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from regex import F
import matplotlib.pyplot as plt

# Gets user input to see how many and which companies to analyze
def get_tickers():
    n = int(input("How many companies would you like to see?"))
    tickers = []
    for i in range(n):
        tickers.append(input("Input a company's stock symbol"))
    return tickers

# Scrapes the site for related articles to the stock
def get_table(news_tables, tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    for ticker in tickers:
        url = finviz_url + ticker
        headers = {'user-agent': 'my-app'}
        page = Request(url = url, headers = headers)
        response = urlopen(page)

        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id = 'news-table')
        news_tables[ticker] = news_table
    return news_tables

# Parses through the given table to get article titles and when it was relased
def get_parsed_data(news_tables, parsed_data):
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
            parsed_data.append([ticker, date, time, title])
    return parsed_data

def main():
    news_tables = {}
    parsed_data = []
    
    get_table(news_tables, get_tickers())
    get_parsed_data(news_tables, parsed_data)

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

    # Conducts sentiment analysis on the title and date of articles and adds its compound sentiment into the dataframe
    vader = SentimentIntensityAnalyzer()
    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date

    # Plots all ticker's sentiment analysis onto a graph
    mean_df = df.groupby(['ticker', 'date']).mean()
    mean_df = mean_df.unstack()
    mean_df = mean_df.xs('compound', axis = 'columns').transpose()
    mean_df.plot(kind = 'bar')

    plt.show()
main()