import sys
import json
import requests
from typing import List, Dict, Optional
from pathlib import Path


from src.config import TICKERS_FILE, USER_AGENT, DATA_DIR


def get_company_tickers() -> Dict[str, Dict]:
    """
    Load ticker-to-company info mapping from a local JSON file.
    """
    p = Path(TICKERS_FILE)
    if not p.exists():
        raise FileNotFoundError(f"Ticker file not found: {p}")
    return json.loads(p.read_text())


def get_filings_for_cik(cik: str) -> Dict:
    """
    Fetch the submissions JSON for a given zero-padded 10-digit CIK.
    """
    headers = {'User-Agent': USER_AGENT}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def extract_10k_urls(submissions: Dict) -> List[str]:
    """
    Extract 10-K document URLs from the submissions JSON.
    """
    recent = submissions['filings']['recent']
    cik = submissions['cik'].lstrip('0')
    urls: List[str] = []
    for form, acc, doc in zip(
        recent['form'],
        recent['accessionNumber'],
        recent['primaryDocument']
    ):
        if form.upper() == '10-K':
            acc_n = acc.replace('-', '')
            urls.append(
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_n}/{doc}"
            )
    return urls


# def download_filings(
#     urls: List[str],
#     out_dir: Path = raw_dir,
#     limit: Optional[int] = None
# ) -> None:
#     """
#     Download each URL into the raw directory.

#     Args:
#         urls: List of document URLs to download.
#         out_dir: Destination directory.
#         limit: Maximum number of files to download (None for all).
#     """
#     headers = {'User-Agent': USER_AGENT}
#     to_download = urls if limit is None else urls[:limit]
#     for url in to_download:
#         fname = url.split('/')[-1]
#         dst = out_dir / fname
#         if dst.exists():
#             print(f"Skipping existing: {fname}")
#             continue
#         print(f"Downloading: {fname}")
#         resp = requests.get(url, headers=headers)
#         resp.raise_for_status()
#         dst.write_bytes(resp.content)


def get_10k_urls_for_ticker(
    ticker: str,
    latest: Optional[int] = None,
    save_to: Optional[Path] = None
) -> List[str]:
    """
    End-to-end helper: given a ticker symbol, return a list of its recent 10-K URLs,
    optionally restricted to the N most recent filings and saved to a file.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        latest: If provided, return only the `latest` number of filings (1 for the single most recent).
        save_to: Optional Path where the URL list will be written (one URL per line).

    Returns:
        List of 10-K document URLs, ordered newest → oldest.
    """
    ticker = ticker.upper()
    tickers = get_company_tickers()
    try:
        entry = next(v for v in tickers.values() if v['ticker'] == ticker)
    except StopIteration:
        raise ValueError(f"Ticker '{ticker}' not found in local mapping.")

    cik = f"{int(entry['cik_str']):010d}"
    subs = get_filings_for_cik(cik)
    urls = extract_10k_urls(subs)

    # If `latest` is specified, take just the first N URLs
    if latest is not None:
        urls = urls[:latest]

    # Optionally save to disk
    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text("\n".join(urls))

    return urls

if __name__ == '__main__':
    get_10k_urls_for_ticker('AAPL',save_to=Path(DATA_DIR) / '10k_urls.txt')

