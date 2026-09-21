import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://www.tn.gov.in/"
SCHEME_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

OUTPUT_FILE = "data/raw/schemes.json"


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
    """Get all scheme detail links from the scheme list page."""

    response = requests.get(
        SCHEME_LIST_URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    scheme_links = []

    for link in soup.find_all("a"):

        href = link.get("href")
        title = link.get_text(" ", strip=True)

        if not href or not title:
            continue

        if "scheme_details.php" in href:

            full_url = urljoin(
                BASE_URL,
                href
            )

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

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    fields = {
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

    scheme = {
        "url": url
    }

    for field_name, html_label in fields.items():

        scheme[field_name] = extract_field(
            soup,
            html_label
        )

    return scheme


def main():

    print("Getting scheme links...")

    scheme_links = get_scheme_links()

    print(
        f"Total schemes found: {len(scheme_links)}"
    )

    schemes = []

    for index, scheme_link in enumerate(
        scheme_links,
        start=1
    ):

        print(
            f"\n[{index}/{len(scheme_links)}] "
            f"{scheme_link['title']}"
        )

        try:

            scheme = extract_scheme(
                scheme_link["url"]
            )

            schemes.append(scheme)

            print("  ✓ Extracted")

        except Exception as e:

            print(
                f"  ✗ Error: {e}"
            )

        # Small delay between requests
        time.sleep(0.5)


    # Save the scraped data
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schemes,
            file,
            ensure_ascii=False,
            indent=4
        )


    print("\n" + "=" * 70)

    print(
        f"Successfully saved {len(schemes)} schemes"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()