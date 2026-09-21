import requests
from bs4 import BeautifulSoup

url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

response = requests.get(url, timeout=30)

print("Status code:", response.status_code)
print("Content length:", len(response.text))

soup = BeautifulSoup(response.text, "html.parser")

print("\nPage title:")
print(soup.title.get_text(strip=True) if soup.title else "No title")

print("\nSearching links for scheme-related content...\n")

links = soup.find_all("a")

for link in links:
    text = link.get_text(" ", strip=True)
    href = link.get("href")

    if "scheme" in text.lower() or "scheme" in str(href).lower():
        print(f"TEXT: {text}")
        print(f"HREF: {href}")
        print("-" * 60)