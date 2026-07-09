"""
Moduł odpowiedzialny za semantyczną ekstrakcję terminologii dziedzinowej
z plików PDF przy użyciu lokalnego modelu językowego Llama 3.2 (Ollama).
"""

import ollama
import time
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Dokonuje ekstrakcji surowego tekstu ze wszystkich stron wskazanego pliku PDF.
    
    Args:
        pdf_path (str): Fizyczna ścieżka do dokumentu PDF na dysku.
        
    Returns:
        str: Scalony tekst ze wszystkich stron dokumentu.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def generate_keywords_from_pdf(model_name: str, pdf_path: str) -> str:
    """
    Analizuje tekst z pliku PDF i generuje listę 20 specjalistycznych słów kluczowych.
    
    Args:
        model_name (str): Nazwa modelu zarejestrowanego w Ollama (np. 'llama3.2').
        pdf_path (str): Ścieżka do pliku PDF stanowiącego bazę wiedzy RAG.
        
    Returns:
        str: Słowa kluczowe oddzielone przecinkami (format initial_prompt).
    """
    pdf_content = extract_text_from_pdf(pdf_path)
    
    system_prompt = (

        "Twoim JEDYNYM zadaniem jest ekstrakcja 20 najtrudniejszych pojęć kontekstowych z podanego tekstu. "
    )

    response = ollama.generate(
        model=model_name,
        system=system_prompt,
        prompt=f"Oto tekst prezentacji do analizy strukturalnej:\n\n{pdf_content}",
        options={
            "temperature": 0.1,   
            "num_ctx": 4096,      
            "num_predict": 150    
        }
    )
    
    return response['response'].strip()


if __name__ == "__main__":
    nazwa_modelu = "llama3.2"
    

    sciezka_do_pdf = "C:/Users/Karol Stachura/Desktop/Projekt_Magisterski/pdf_files/Lab2 - Wstęp do klasyfikacji.pdf" 
    
    print(f"Uruchamiam Eksperyment RAG 2 (Model: {nazwa_modelu})...")
    print(f"Analiza semantyczna pliku: {sciezka_do_pdf}")
    
    start_time = time.time()
    
    try:
        wynik = generate_keywords_from_pdf(nazwa_modelu, sciezka_do_pdf)
        end_time = time.time()
        
        print(f"\nWygenerowany initial_prompt dla Whispera (Benchmark PDF):\n{wynik}")
        print(f"\n--- CZAS OPERACJI OLLAMA (Llama 3.2): {end_time - start_time:.2f} sekund ---")
        
    except Exception as e:
        print(f"\nBŁĄD PROCESU: {e}")
        print("Upewnij się, że usługa Ollama jest aktywna oraz ścieżka do pliku PDF jest prawidłowa.")