"""
Medical Text Preprocessing Module
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class MedicalTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Медицинские термины, которые НЕ нужно удалять
        self.medical_terms = {'patient', 'diagnosis', 'treatment', 
                              'symptom', 'disease', 'therapy', 'clinical'}
        self.stop_words = self.stop_words - self.medical_terms
    
    def clean_text(self, text):
        """1. Очистка медицинского текста"""
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', '', text)
        # Удаление специальных символов, кроме медицинских
        text = re.sub(r'[^a-zA-Z0-9\s.,;:()-]', '', text)
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text):
        """2. Токенизация"""
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens):
        """3. Удаление НЕ-медицинских стоп-слов"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """4. Лемматизация медицинских терминов"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text):
        """Полный пайплайн обработки"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize(filtered)
        return {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'filtered_tokens': filtered,
            'lemmatized': lemmatized,
            'processed_text': ' '.join(lemmatized)
        }