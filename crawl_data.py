import requests
from bs4 import BeautifulSoup

URLS = [
    "https://tuanlda78202.github.io/",              
    "https://tuanlda78202.github.io/repositories/", 
    "https://tuanlda78202.github.io/cv/"            
]

def fetch(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def texts_from(soup):
    
    for tag in soup.select("h1, h2, h3, p, li, blockquote, code, pre, figcaption"):
        t = " ".join(tag.get_text(" ", strip=True).split())
        if t:
            yield t

for url in URLS:
    print("="*50)
    print("Trang:", url)
    try:
        soup = fetch(url)
        found = False
        for t in texts_from(soup):
            print("•", t)
            found = True
        if not found:
            print("(Không thấy nội dung văn bản ở các thẻ cơ bản)")
    except Exception as e:
        print("Lỗi:", e)

