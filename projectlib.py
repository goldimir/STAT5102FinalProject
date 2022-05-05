#As there are many cryptocurrencies nowaday, it is very easy that two Altcoins with the same name. 
#Therefore, we need to see the coin info to confirm it is what we want
#And the Coin ID can search in https://docs.google.com/spreadsheets/d/1wTTuxXt8n9q7C4NDXqQpI3wpKu1_5bGVmP9Xz0XGSyU/edit#gid=0

import requests
def get_coin_info(Id, result = False):
    url = f'https://api.coingecko.com/api/v3/coins/{Id}?localization=false&tickers=false'
    
    r = requests.get(url)
    j= r.json()
    
    Symbol = j['symbol']
    Name = j['name']
    categories = j['categories']
    description = j['description']["en"]
    
    print(f"Symbol of coin {Symbol}")
    print(f"Name of coin {Name}")
    for category in categories:
        print(f"{Name} is related to {category}")
        
    print("Description:")
    print(f"{description}")
    
    if result:
        return j
#Get historical market data include price, market cap, and 24h volume (granularity auto)
#Minutely data will be used for duration within 1 day, Hourly data will be used for duration between 1 day and 90 days, Daily data will be used for duration above 90 days.

def get_coin_historical_data(Id,days='max',vs_currency='usd',interval='daily'):
    url = f'https://api.coingecko.com/api/v3/coins/{Id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}'
    
    #get request and read into json
    r = requests.get(url)
    j= r.json()
    
    
    return j