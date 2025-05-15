import requests

TOKEN = '7544529562:AAHiLMgvUXanJDWHm0xFOsIfbasBgEqOaQw'
URL = f'https://api.telegram.org/bot{TOKEN}/getUpdates'

response = requests.get(URL)
print(response.json())