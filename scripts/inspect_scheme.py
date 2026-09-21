import requests
from bs4 import BeautifulSoup


def extract_field(soup, field_name):
    """
    Find a field in a scheme detail page and return its cleaned value.
    """

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

    # Normalize empty values
    if not value:
        return None

    return value


url = "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng=="

response = requests.get(url, timeout=30)

print("Status code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


fields_to_extract = {
    "title": "Scheme Title/Name:",
    "department": "Concerned Department:",
    "district": "Concerned District:",
    "organisation": "Organisation Name:",
    "associated_scheme": "Associated Scheme:",
    "sponsored_by": "Sponsered By:",
    "funding_pattern": "Funding Pattern:",
    "beneficiaries": "Beneficiaries:",
    "benefit_type": "Types of Benefits:",
    "eligibility": "Eligibility criteria:",
    "how_to_avail": "How To avail:",
    "validity": "Validity of the Scheme:",
    "introduced_on": "Introduced On:",
    "description": "Description:",
    "scheme_type": "Scheme Type:",
    "uploaded_file": "Uploaded File:",
}


scheme = {}

for field_name, html_label in fields_to_extract.items():
    scheme[field_name] = extract_field(soup, html_label)


print("\nStructured Scheme Data")
print("=" * 60)

for key, value in scheme.items():
    print(f"{key}: {value}")