import asyncio
import json
import os
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

async def main():
    # Construct the file URL for the local HTML file
    url = 'https://peraturan.go.id/id/uu-no-1-tahun-2026'
    schema = {
        "name": "Undang-Undang",
        "baseSelector": "section#description",
        "fields": [
            {
                "name": "judul", 
                "selector": "div.detail_title_1", 
                "type": "text"
            },
            {
                "name": "jenis", 
                "selector": "tbody tr:nth-child(1) td", 
                "type": "text"
            },
            {
                "name": "pemrakarsa", 
                "selector": "tbody tr:nth-child(2) td", 
                "type": "text"
            },
            {
                "name": "nomor", 
                "selector": "tbody tr:nth-child(3) td", 
                "type": "text"
            },
            {
                "name": "tahun", 
                "selector": "tbody tr:nth-child(4) td", 
                "type": "text"
            },
            {
                "name": "tentang", 
                "selector": "tbody tr:nth-child(5) td", 
                "type": "text"
            },
            {
                "name": "tempat_penetapan", 
                "selector": "tbody tr:nth-child(6) td", 
                "type": "text"
            },
            {
                "name": "ditetapkan_tanggal", 
                "selector": "tbody tr:nth-child(7) td", 
                "type": "text"
            },
            {
                "name": "pejabat yang menetapkan", 
                "selector": "tbody tr:nth-child(8) td", 
                "type": "text"
            },
            {
                "name": "status", 
                "selector": "tbody tr:nth-child(9) td", 
                "type": "text"
            },
            {
                "name": "dokumen_peraturan", 
                "selector": "tbody tr:nth-child(10) td a", 
                "type": "attribute",
                "attribute": "href"
            }
        ]
    }

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=JsonCssExtractionStrategy(schema)
            )
        )
        
        if result.success:
            # The JSON output is stored in 'extracted_content'
            data = json.loads(result.extracted_content)
            # Print the formatted JSON output
            print(json.dumps(data, indent=2))
            
            with open('uu.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            print(f"Extraction failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
