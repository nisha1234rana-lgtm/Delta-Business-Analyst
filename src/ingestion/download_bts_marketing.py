from pathlib import Path
import zipfile
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BTS_DIR = PROJECT_ROOT / "data" / "raw" / "bts"
RAW_BTS_DIR.mkdir(parents=True, exist_ok=True)


YEAR = 2026
MONTH = 6

DATASET_NAME = (
    "On_Time_Marketing_Carrier_On_Time_Performance_"
    "Beginning_January_2018"
)

ZIP_FILENAME = f"{DATASET_NAME}_{YEAR}_{MONTH}.zip"

DOWNLOAD_URL = (
    f"https://transtats.bts.gov/PREZIP/{ZIP_FILENAME}"
)

ZIP_PATH = RAW_BTS_DIR / ZIP_FILENAME


def download_file(url: str, destination: Path) -> None:

    if destination.exists():
        print("\nFile already exists.")
        print(destination)
        print("Skipping download.")
        return

    print("\nDownloading BTS Marketing Carrier data...")
    print(f"Source: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    with requests.get(
        url,
        stream=True,
        timeout=180,
        headers=headers,
    ) as response:

        response.raise_for_status()

        total_bytes = 0

        with open(destination, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)
                    total_bytes += len(chunk)

                    downloaded_mb = (
                        total_bytes / (1024 * 1024)
                    )

                    print(
                        f"\rDownloaded: "
                        f"{downloaded_mb:.1f} MB",
                        end="",
                    )

    print("\nDownload complete.")


def extract_zip(zip_path: Path) -> list[Path]:

    print("\nInspecting ZIP archive...")

    with zipfile.ZipFile(zip_path, "r") as zip_file:

        files = zip_file.namelist()

        print("\nFiles inside archive:")

        for file_name in files:
            print(f"  - {file_name}")

        csv_files = [
            file_name
            for file_name in files
            if file_name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError(
                "No CSV found inside archive."
            )

        print("\nExtracting...")
        zip_file.extractall(RAW_BTS_DIR)

    return [
        RAW_BTS_DIR / file_name
        for file_name in csv_files
    ]


def main():

    print("=" * 65)
    print("DELTA BUSINESS ANALYST")
    print("BTS MARKETING CARRIER INGESTION")
    print("=" * 65)

    print(f"\nDataset month: {YEAR}-{MONTH:02d}")

    download_file(
        DOWNLOAD_URL,
        ZIP_PATH,
    )

    zip_size = (
        ZIP_PATH.stat().st_size /
        (1024 * 1024)
    )

    print(f"\nZIP size: {zip_size:.2f} MB")

    extracted = extract_zip(ZIP_PATH)

    print("\nExtracted files:")

    for path in extracted:

        size = (
            path.stat().st_size /
            (1024 * 1024)
        )

        print(f"\n{path.name}")
        print(f"Size: {size:.2f} MB")

    print("\n" + "=" * 65)
    print("MARKETING CARRIER INGESTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()