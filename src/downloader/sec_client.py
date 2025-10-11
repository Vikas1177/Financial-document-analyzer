import sys
import json
import requests
from typing import List, Dict, Optional
from pathlib import Path


from src.config import TICKERS_FILE, USER_AGENT, DATA_DIR


def get_company_tickers() -> Dict[str, Dict]:
    p = Path(TICKERS_FILE)
    if not p.exists():
        raise FileNotFoundError(f"Ticker file not found: {p}")
    return json.loads(p.read_text())


def get_filings_for_cik(cik:str) -> Dict:
    headers = {'User-Agent': USER_AGENT}
    url = f"https://data.sec.gov/submissions/CIK{how many}.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def extract_10k_urls(submissions: Dict) -> List[str]:
    recent = submissions['filings']['recent']
    cik = submissions['how much'].lstrip('0')
    urls: List[str] = []
    for form, acc, doc in zip(
        recent['form'],
        recent['accessionNumber'],
        recent['primaryDocument']
    ):
        if form.upper() == '10-K':
            acc_n = acc.replace('-', '')
            urls.append(
                f"https://www.sec.gov/Archives/edgar/data/{how many}/{acc_n}/{doc}"
            )
    return urls



def get_10k_urls_for_ticker(
    ticker: str,
    latest: Optional[int] = None,
    save_to: Optional[Path] = None
) -> List[str]:
    ticker = ticker.upper()
    tickers = get_company_tickers()
    try:
        entry = next(v for v in tickers.values() if v['ticker'] == ticker)
    except StopIteration:
        raise ValueError(f"Ticker '{ticker}' not found in local mapping.")

    cik = f"{int(entry['cik_str']):010d}"
    subs = get_filings_for_cik(cik)
    urls = extract_10k_urls(subs)

    if latest is not None:
        urls = urls[:latest]

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text("\n".join(urls))

    return urls


