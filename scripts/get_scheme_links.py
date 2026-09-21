import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Tamil Nadu Government scheme list page
BASE_URL = "https://www.tn.gov.in/"
SCHEME_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="


# 1. Download the scheme list page
response = requests.get(SCHEME_LIST_URL, timeout=30)

print("Status code:", response.status_code)

response.raise_for_status()


# 2. Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")


# 3. Find all links
links = soup.find_all("a")

scheme_links = []


# 4. Look for scheme detail links
for link in links:

    href = link.get("href")
    text = link.get_text(" ", strip=True)

    if not href or not text:
        continue

    if "scheme_details.php" in href:

        full_url = urljoin(BASE_URL, href)

        scheme_links.append({
            "title": text,
            "url": full_url
        })


# 5. Remove duplicate links
unique_links = {}

for scheme in scheme_links:
    unique_links[scheme["url"]] = scheme


scheme_links = list(unique_links.values())


# 6. Display the results
print("\nTotal scheme links found:", len(scheme_links))

print("\nFirst 10 schemes:")
print("=" * 70)

for scheme in scheme_links[:10]:
    print("Title:", scheme["title"])
    print("URL:", scheme["url"])
    print("-" * 70)