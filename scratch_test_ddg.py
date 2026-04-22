import requests
import bs4

def test_ddg(query):
    url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 202:
        print("Blocked (202 Accepted means anti-bot JS challenge)")
    elif 'result' in resp.text.lower():
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all('td', class_='result-snippet')
        print(f"Found {len(results)} results")
        for idx, result in enumerate(results[:3]):
            print(f"[{idx}] {result.text.strip()}")
    else:
        print("No results found or blocked without 202")
        print(resp.text[:500])

if __name__ == "__main__":
    test_ddg("Saketh Poduvu")
