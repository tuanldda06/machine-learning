import requests
from bs4 import BeautifulSoup

url = "https://tuanlda78202.github.io/"
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')


print("="*50)

paragraphs = soup.find_all('p')
for p in paragraphs:
    text = p.get_text(strip=True)  
    if text:  
        print("•", text)