import asyncio
import os
import sys
import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Set default encoding to UTF-8 for Windows console output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def download_with_requests(url, output_path):
    """
    Fallback method using requests to download a file if Crawl4AI 
    doesn't capture the redirected download automatically.
    """
    try:
        # allow_redirects=True is default, but we'll be explicit
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Check if the response is actually a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, content_type
    except Exception as e:
        return False, str(e)

async def download_kpu_pdfs_final(start_num: int, end_num: int):
    """
    Enhanced download script that follows redirects and captures the final PDF.
    """
    base_url = "https://jdih.kpu.go.id/undang-undang/download"
    output_dir = os.path.join(os.getcwd(), "kpu_downloads_final")
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print(f"Starting Final PDF Download (Range: {start_num} to {end_num})")
    print(f"Output Directory: {output_dir}")
    print("="*60)

    # Note: If Crawl4AI's accept_downloads is failing due to how the site
    # triggers the download (e.g., via a meta-refresh or specific JS),
    # we use requests as a reliable fallback for direct file streams.

    for i in range(start_num, end_num + 1):
        url = f"{base_url}/{i}"
        file_name = f"kpu_undang_undang_{i}.pdf"
        output_path = os.path.join(output_dir, file_name)
        
        print(f"Processing [{i}]: {url}")
        
        # First, try to see where it redirects
        try:
            success, info = download_with_requests(url, output_path)
            if success:
                print(f"  [OK] Saved: {file_name} (Content-Type: {info})")
            else:
                print(f"  [FAIL] Failed to download {i}: {info}")
        except Exception as e:
            print(f"  [ERROR] Error processing {i}: {e}")

    print("="*60)
    print(f"Process completed.")
    print("="*60)

if __name__ == "__main__":
    # Download range 1 to 10
    try:
        asyncio.run(download_kpu_pdfs_final(1, 10))
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
