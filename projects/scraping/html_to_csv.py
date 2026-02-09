import os
import sys
import pandas as pd
from bs4 import BeautifulSoup

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    html_file = os.path.join(script_dir, "html.html")
    csv_file = os.path.join(script_dir, "csv.csv")
    
    # Check input file existence
    if not os.path.exists(html_file):
        print(f"Error: File not found at {html_file}")
        return

    print(f"Reading HTML file: {html_file}")
    
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return

    # Find the data table
    table = soup.find("table")
    if not table:
        print("Error: No <table> tag found in the HTML.")
        return

    # Extract Headers
    headers = []
    # Try finding headers in <thead> first
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            headers = [th.get_text(separator=" ", strip=True) for th in header_row.find_all(["th", "td"])]
    
    # If not found in thead, try the first row of the table
    if not headers:
        first_row = table.find("tr")
        if first_row:
            headers = [th.get_text(separator=" ", strip=True) for th in first_row.find_all(["th", "td"])]
    
    print(f"Headers detected ({len(headers)}): {headers}")

    # Extract Data Rows
    data = []
    tbody = table.find("tbody")
    
    # If tbody exists, iterate its rows; otherwise iterate all table rows
    rows_to_scan = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows_to_scan:
        # Find all cells in the row
        cells = row.find_all(["td", "th"])
        
        # skip empty rows
        if not cells:
            continue
            
        # Extract text clearly, separating internal tags (like <br>) with spaces
        row_data = [cell.get_text(separator=" ", strip=True) for cell in cells]
        
        # If we didn't use tbody, we might re-encounter the header row. Skip it if it matches headers.
        if not tbody and row_data == headers:
            continue
            
        data.append(row_data)

    print(f"Extracted {len(data)} rows of data.")

    # Create DataFrame
    # Note: If data rows have different lengths than headers, pandas handles it (though it might align poorly)
    try:
        if headers:
            df = pd.DataFrame(data, columns=headers)
        else:
            df = pd.DataFrame(data)
            
        # Save to CSV
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"Successfully saved to: {csv_file}")
        
        # Display first few rows
        print("\nPreview:")
        print(df.head())
        
    except Exception as e:
        print(f"Error creating/saving DataFrame: {e}")

if __name__ == "__main__":
    main()
