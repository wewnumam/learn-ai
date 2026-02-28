import os
from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=".env")

username = os.getenv("PROXY_USERNAME")
password = os.getenv("PROXY_PASSWORD")
country = os.getenv("PROXY_COUNTRY")
proxy = os.getenv("PROXY_ENTRY")
ports = ['8000', '8001', '8002', '8003', '8004', '8005']

for port in ports:
   proxies = {
      "https": ('https://user-%s-country-%s:%s@%s:%s' % (username, country, password, proxy, port))
   }
      
   response=requests.get("https://ip.oxylabs.io/location", proxies=proxies)

   print(response.content)