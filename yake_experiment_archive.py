import pdfplumber
import yake
import time

def extract_keywords_from_pdf(pdf_path, top_n=30):
    text_content = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + " "

    custom_kw_extractor = yake.KeywordExtractor(
        lan="pl",
        n=2,
        dedupLim=0.9,
        top=top_n,
        features=None
    )

    keywords = custom_kw_extractor.extract_keywords(text_content)

    top_words = [kw[0] for kw in keywords]
    
    return ", ".join(top_words)

if __name__ == "__main__":
    start_time = time.time()
    
    prompt_dla_whispera = extract_keywords_from_pdf("pdf_files/Lab2 - Wstęp do klasyfikacji.pdf")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Wstrzykujemy do Whispera: {prompt_dla_whispera}")
    print(f"\n--- CZAS EKSTRAKCJI YAKE: {execution_time:.2f} sekund ---")

# TEST
prompt_dla_whispera = extract_keywords_from_pdf("pdf_files/Lab2 - Wstęp do klasyfikacji.pdf")
print(f"Wstrzykujemy do Whispera: {prompt_dla_whispera}")