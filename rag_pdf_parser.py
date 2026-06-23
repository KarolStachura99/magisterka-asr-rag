import pdfplumber
import re
from collections import Counter

# Twarda lista słów wykluczonych (Stop Words)
# Zbiór (set) zapewnia błyskawiczne wyszukiwanie w czasie O(1)
STOP_WORDS = {
    "jest", "oraz", "albo", "może", "tylko", "które", "tego", "tych", "przez", 
    "przy", "jako", "jeśli", "żeby", "więc", "także", "bardzo", "często", "będą",
    "różnych", "dużych", "liczba", "wartości", "cech", "modelu", "zbiorów", "wyników",
    "informatyki", "danych", "analizy", "statystycznej", "metod", "algorytmów", "klasyfikacji",
    "this", "that", "there", "with", "from", "have"
}

def extract_keywords_from_pdf(pdf_path, top_n=30):
    text_content = ""
    
    # 1. Odczyt PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + " "

    # 2. Inteligentna ekstrakcja (Regex)
    # \b[A-Z]{2,}\b  -> Wyłapuje akronimy (np. AI, SVM, NLP, API)
    # \b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{4,}\b -> Wyłapuje standardowe słowa (min. 4 litery)
    pattern = r'\b[A-Z]{2,}\b|\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{4,}\b'
    raw_words = re.findall(pattern, text_content)

    # 3. Filtrowanie Stop Words z zachowaniem oryginalnej wielkości liter
    filtered_words = []
    for word in raw_words:
        # Zamieniamy na małe litery tylko na potrzeby sprawdzenia w słowniku
        if word.lower() not in STOP_WORDS:
            filtered_words.append(word)

    # 4. Zliczanie najczęstszych pojęć
    word_counts = Counter(filtered_words)
    
    # Zwracamy czysty string gotowy dla parametru initial_prompt Whispera
    top_words = [word for word, count in word_counts.most_common(top_n)]
    return ", ".join(top_words)

prompt_dla_whispera = extract_keywords_from_pdf("pdf_files/Lab2 - Wstęp do klasyfikacji.pdf")
print(f"Wstrzykujemy do Whispera: {prompt_dla_whispera}")