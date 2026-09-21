import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://www.tn.gov.in/"
SCHEME_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="


def extract_field(soup, field_name):
    """Extract the value of a field from a scheme detail page."""

    label = soup.find(
        string=lambda text: text and field_name in text
    )

    if not label:
        return None

    row = label.find_parent("tr")

    if not row:
        return None

    cells = row.find_all("td")

    if len(cells) < 2:
        return None

    value = cells[1].get_text(" ", strip=True)

    if not value:
        return None

    return value


def get_scheme_links():
    """Get scheme detail links from the scheme list page."""

    response = requests.get(
        SCHEME_LIST_URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    scheme_links = []

    for link in soup.find_all("a"):

        href = link.get("href")
        title = link.get_text(" ", strip=True)

        if not href or not title:
            continue

        if "scheme_details.php" in href:

            full_url = urljoin(BASE_URL, href)

            scheme_links.append({
                "title": title,
                "url": full_url
            })

    # Remove duplicate URLs
    unique_links = {}

    for scheme in scheme_links:
        unique_links[scheme["url"]] = scheme

    return list(unique_links.values())


def extract_scheme(url):
    """Extract structured information from one scheme page."""

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    fields = {
        "title": "Scheme Title/Name:",
        "department": "Concerned Department:",
        "beneficiaries": "Beneficiaries:",
        "benefit_type": "Types of Benefits:",
        "funding_pattern": "Funding Pattern:",
        "description": "Description:",
    }

    scheme = {
        "url": url
    }

    for field_name, html_label in fields.items():

        scheme[field_name] = extract_field(
            soup,
            html_label
        )

    return scheme


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

print("Getting scheme links...")

scheme_links = get_scheme_links()

print("Total schemes found:", len(scheme_links))


# Test only the first 3 schemes
test_schemes = scheme_links[:3]


print("\nTesting first 3 schemes")
print("=" * 80)


for index, scheme_link in enumerate(test_schemes, start=1):

    print(f"\nScheme {index}")
    print("-" * 80)

    print("List title:", scheme_link["title"])
    print("URL:", scheme_link["url"])

    try:

        scheme = extract_scheme(
            scheme_link["url"]
        )

        print("\nExtracted data:")

        print("Title:", scheme["title"])
        print("Department:", scheme["department"])
        print("Beneficiaries:", scheme["beneficiaries"])
        print("Benefit type:", scheme["benefit_type"])
        print("Funding pattern:", scheme["funding_pattern"])
        print("Description:", scheme["description"])

    except Exception as e:

        print("ERROR:", e)