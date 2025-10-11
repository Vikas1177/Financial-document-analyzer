import sys
import json
from pathlib import Path
from typing import List, Optional, Union

from src.config import SEC_API, DATA_DIR
from sec_api import ExtractorApi


def load_urls_from_file(file_path: Union[Path, str]) -> List[str]:
    """Load URLs from a text file, with error handling."""
    url_file = Path(file_path)
    
    if not url_file.exists():
        raise FileNotFoundError(f"URL file not found: {url_file}")
    
    try:
        urls = [url.strip() for url in url_file.read_text(encoding='utf-8').splitlines() if url.strip()]
        print(f"Loaded {len(urls)} URL(s) from {url_file}")
        return urls
    except Exception as e:
        raise RuntimeError(f"Failed to read URLs from {url_file}: {e}")

def _save_results(
    results: List[str], 
    save_to: Union[Path, str], 
    output_format: str,
    indices: List[int]
) -> None:
    """Helper function to save extraction results."""
    save_path = Path(save_to)
    
    # If save_to is a directory, create appropriate filename
    if save_path.is_dir() or (not save_path.suffix and not save_path.exists()):
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"risk_factors_filings.txt"
        save_path = save_path / filename
    else:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save as plain text with separators
        content_parts = []
        for idx, text in zip(indices, results):
            content_parts.append(f"=== FILING #{idx} RISK FACTORS ===\n")
            content_parts.append(text if text else "[No content extracted]")
        
        content = "\n".join(content_parts)
        save_path.write_text(content, encoding='utf-8')
        
        print(f"✓ Saved {len(results)} risk factor text(s) to {save_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
        raise


def extract_risk_factors(
    urls: List[str],
    select: Optional[List[int]] = None,
    save_to: Optional[Union[Path, str]] = None,
    output_format: str = 'txt'
) -> List[str]:
    """
    Extract Item 1A (Risk Factors) text from selected 10-K filings.

    Args:
        urls: List of SEC EDGAR 10-K filing URLs.
        select: List of indices to extract (0-based). If None, extracts all.
        save_to: Optional path/directory to save results. If directory, auto-generates filename.
        output_format: Format to save ('txt' or 'json'). Default is 'txt'.

    Returns:
        List of extracted Risk Factors texts.
    
    Raises:
        ValueError: If invalid indices are provided or URLs list is empty.
        FileNotFoundError: If save_to directory doesn't exist and can't be created.
    """
    if not urls:
        raise ValueError("URLs list cannot be empty")
    
    # Validate select indices
    if select is not None:
        invalid_indices = [i for i in select if i < 0 or i >= len(urls)]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Valid range: 0-{len(urls)-1}")
    
    extractor = ExtractorApi(SEC_API)
    results: List[str] = []
    
    # Determine which URLs to process
    indices_to_process = select if select is not None else list(range(len(urls)))
    urls_to_process = [urls[i] for i in indices_to_process]
    
    print(f"Processing {len(urls_to_process)} filing(s)...")
    
    for idx, url in zip(indices_to_process, urls_to_process):
        try:
            print(f"Extracting Risk Factors from filing #{idx}: {url}")
            text = extractor.get_section(url, '1A', 'text')
            if text:
                results.append(text.strip())
                print(f"✓ Successfully extracted {len(text)} characters")
            else:
                print("⚠️ No Risk Factors section found")
                results.append("")
        except Exception as e:
            print(f"⚠️ Failed to extract from filing #{idx}: {e}")
            results.append("")
    
    # Save results if path is provided
    if save_to:
        _save_results(results, save_to, output_format, indices_to_process)
    
    return results



if __name__ == '__main__':
    try:
        # Load URLs from file
        url_file = Path(DATA_DIR) / '10k_urls.txt'
        urls = load_urls_from_file(url_file)
        
        if not urls:
            print("No URLs found in file. Exiting.")
            sys.exit(1)
        

        # # Example 2: Extract the second filing and save to directory (auto-filename)
        # print("\n=== Extracting Second Filing ===")
        # if len(urls) > 1:
        #     second_text = extract_risk_factors(
        #         urls, 
        #         select=[1], 
        #         save_to=Path(DATA_DIR)  # Will auto-generate filename
        #     )
        # else:
        #     print("Only one URL available, skipping second filing extraction.")

        available_indices = list(range(2))
        
        # Example 3: Extract multiple filings and save as JSON
        print("\n=== Extracting Multiple Filings ===")
        multiple_texts = extract_risk_factors(
            urls, 
            select=available_indices,
            save_to=Path(DATA_DIR) 
        )
        
        print(f"\n✓ Successfully processed {len(urls)} total URL(s)")
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        sys.exit(1)
