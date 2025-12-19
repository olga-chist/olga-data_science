"""
Medical PDF Loader and Processor
"""

import PyPDF2
from pathlib import Path
from .nlp_preprocessor import MedicalTextPreprocessor

class MedicalPDFLoader:
    def __init__(self, knowledge_base_path="medical_knowledge"):
        self.knowledge_base = Path(knowledge_base_path)
        self.preprocessor = MedicalTextPreprocessor()
        self.documents = []  # [{text, metadata}, ...]
        
    def load_all_documents(self):
        """Загружает все PDF и текстовые файлы из папки"""
        if not self.knowledge_base.exists():
            print(f"⚠️ Папка {self.knowledge_base} не найдена")
            return []
        
        print(f"📁 Загрузка документов из: {self.knowledge_base}")
        
        # Загружаем PDF файлы
        pdf_files = list(self.knowledge_base.glob("*.pdf"))
        for pdf_path in pdf_files:
            self._load_pdf(pdf_path)
        
        # Загружаем текстовые файлы
        txt_files = list(self.knowledge_base.glob("*.txt"))
        for txt_path in txt_files:
            self._load_txt(txt_path)
        
        print(f"✅ Загружено документов: {len(self.documents)}")
        return self.documents
    
    def _load_pdf(self, pdf_path):
        """Извлекает текст из PDF"""
        try:
            print(f"  📄 Загружаю PDF: {pdf_path.name}")
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        # Предобработка текста
                        processed = self.preprocessor.process(page_text)
                        text += processed['processed_text'] + "\n\n"
            
            if text.strip():
                self.documents.append({
                    'text': text.strip(),
                    'source': pdf_path.name,
                    'type': 'pdf',
                    'pages': num_pages,
                    'preprocessing_stats': {
                        'original_length': len(text),
                        'processed_length': len(text.strip())
                    }
                })
                
        except Exception as e:
            print(f"  ❌ Ошибка при загрузке {pdf_path.name}: {e}")
    
    def _load_txt(self, txt_path):
        """Загружает текстовые файлы"""
        try:
            print(f"  📝 Загружаю TXT: {txt_path.name}")
            
            with open(txt_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                
                # Предобработка
                processed = self.preprocessor.process(raw_text)
                
                self.documents.append({
                    'text': processed['processed_text'],
                    'source': txt_path.name,
                    'type': 'txt',
                    'preprocessing_stats': {
                        'original_length': len(raw_text),
                        'processed_length': len(processed['processed_text']),
                        'tokens_count': len(processed['lemmatized'])
                    }
                })
                
        except Exception as e:
            print(f"  ❌ Ошибка при загрузке {txt_path.name}: {e}")
    
    def get_statistics(self):
        """Статистика по загруженным документам"""
        if not self.documents:
            return {"total_documents": 0}
        
        total_text_length = sum(len(doc['text']) for doc in self.documents)
        pdf_count = sum(1 for doc in self.documents if doc['type'] == 'pdf')
        txt_count = sum(1 for doc in self.documents if doc['type'] == 'txt')
        
        return {
            'total_documents': len(self.documents),
            'pdf_files': pdf_count,
            'txt_files': txt_count,
            'total_text_characters': total_text_length,
            'avg_text_length': total_text_length // len(self.documents)
        }
    
    def print_sample(self, doc_index=0, max_chars=500):
        """Показывает образец текста из документа"""
        if not self.documents:
            print("Нет загруженных документов")
            return
        
        doc = self.documents[doc_index]
        print(f"\n{'='*60}")
        print(f"📑 Документ #{doc_index + 1}: {doc['source']} ({doc['type']})")
        print(f"{'='*60}")
        print("📄 Текст (первые {} символов):".format(max_chars))
        print("-"*40)
        print(doc['text'][:max_chars] + "...")
        print("-"*40)
        print("📊 Метаданные:", doc.get('preprocessing_stats', {}))