import asyncio
import json
import os
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

async def main():
    # Construct the file URL for the local HTML file
    url = 'https://peraturan.go.id/uu/rekapitulasi'
    schema = {
        "name": "Rekapitulasi Undang-Undang",
        "baseSelector": "div.accordion_2 div.card",
        "fields": [
            {
                "name": "tahun", 
                "selector": "h5.mb-0 a", 
                "type": "text"
            },
            {
                "name": "jumlah_peraturan", 
                "selector": "div.card-body li:nth-child(1) small", 
                "type": "text"
            },
            {
                "name": "berlaku", 
                "selector": "div.card-body li:nth-child(2) small", 
                "type": "text"
            },
            {
                "name": "tidak_berlaku", 
                "selector": "div.card-body li:nth-child(3) small", 
                "type": "text"
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
            
            # Post-process to convert strings to integers
            for item in data:
                for key in ["tahun", "jumlah_peraturan", "berlaku", "tidak_berlaku"]:
                    if key in item and item[key]:
                        try:
                            # Strip whitespace and convert to int
                            item[key] = int(item[key].strip())
                        except (ValueError, TypeError):
                            # Fallback if conversion fails (keep as original or None)
                            pass

            # Print the formatted JSON output
            # Use ensure_ascii=True or redirect output to avoid console encoding issues on Windows
            print(json.dumps(data, indent=2))

            # Use an absolute path for the output file to be safe, or relative to the script
            output_path = Path(__file__).parent / 'rekapitulasi_uu.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Data saved to {output_path}")
        else:
            print(f"Extraction failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
