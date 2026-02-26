import os
from pypdf import PdfReader

INPUT_FOLDER = "pdfs"
OUTPUT_FOLDER = "text_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"\n--- Page {page_num+1} ---\n{page_text}")

        return "\n".join(text)

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def process_folder():
    for filename in os.listdir(INPUT_FOLDER):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(
                OUTPUT_FOLDER,
                filename.replace(".pdf", ".txt")
            )

            print(f"Processing: {filename}")

            text = extract_text_from_pdf(pdf_path)

            if text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)


if __name__ == "__main__":
    process_folder()

    print("All PDFs processed.")