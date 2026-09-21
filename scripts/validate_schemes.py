import json


INPUT_FILE = "data/raw/schemes.json"


def main():

    # 1. Load JSON
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        schemes = json.load(file)


    print("JSON validation")
    print("=" * 70)

    # 2. Check number of schemes
    print("Total schemes:", len(schemes))


    # 3. Check fields
    if schemes:

        print("\nFields in each scheme:")
        print("-" * 70)

        for field in schemes[0].keys():
            print(field)


    # 4. Count missing values
    print("\nMissing values")
    print("-" * 70)

    field_missing_count = {}

    for scheme in schemes:

        for field, value in scheme.items():

            if value is None or value == "":

                field_missing_count[field] = (
                    field_missing_count.get(field, 0) + 1
                )


    for field, count in field_missing_count.items():

        print(
            f"{field}: {count} missing"
        )


    # 5. Show first scheme
    print("\nFirst scheme")
    print("=" * 70)

    print(
        json.dumps(
            schemes[0],
            indent=4,
            ensure_ascii=False
        )
    )


    # 6. Basic validation
    print("\nValidation result")
    print("=" * 70)

    if len(schemes) == 54:

        print("✓ Expected 54 schemes found")

    else:

        print(
            f"⚠ Expected 54 schemes, "
            f"but found {len(schemes)}"
        )


if __name__ == "__main__":
    main()