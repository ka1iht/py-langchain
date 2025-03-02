from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://aws.amazon.com/architecture/"

df = pd.DataFrame()
links = []

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

for link in soup.find_all('a'):
    subpages = str(link.get("href"))
    if "architecture/" in subpages:
        links.append(subpages)

print(links)