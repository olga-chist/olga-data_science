"""
Medical Vector Store - для создания и поиска в векторной базе
"""
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

class MedicalVectorStore:
    def __init__(self, persist_directory="./chroma_medical_db"):
        self.persist_directory = persist_directory
        
        # Используем легкую модель эмбеддингов
        print("🧠 Загружаю модель для эмбеддингов...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Можно поменять на 'cuda' если есть GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Модель загружена")
        
        self.vectorstore = None
    
    def create_from_documents(self, documents):
        """Создает векторную базу из документов"""
        if not documents:
            print("❌ Нет документов для обработки")
            return None
        
        print(f"📚 Начинаю обработку {len(documents)} документов...")
        
        # 1. Подготавливаем тексты
        all_texts = []
        all_metadatas = []
        
        for doc in documents:
            source = doc.get('source', 'unknown')
            doc_type = doc.get('type', 'unknown')
            
            # 2. Разделяем длинные тексты на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,      # Размер чанка
                chunk_overlap=100,   # Перекрытие между чанками
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Разделяем текст документа
            chunks = text_splitter.split_text(doc['text'])
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Игнорируем очень короткие чанки
                    all_texts.append(chunk)
                    all_metadatas.append({
                        'source': source,
                        'type': doc_type,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
        
        print(f"✂️  Создано {len(all_texts)} текстовых фрагментов")
        
        if not all_texts:
            print("❌ Не удалось создать текстовые фрагменты")
            return None
        
        # 3. Создаем векторную базу
        print("⚡ Создаю векторные эмбеддинги...")
        start_time = time.time()
        
        try:
            self.vectorstore = Chroma.from_texts(
                texts=all_texts,
                metadatas=all_metadatas,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Сохраняем на диск
            self.vectorstore.persist()
            
            elapsed = time.time() - start_time
            print(f"✅ Векторная база создана за {elapsed:.1f} сек.")
            print(f"💾 Сохранено в: {self.persist_directory}")
            print(f"📊 Всего фрагментов: {len(all_texts)}")
            
        except Exception as e:
            print(f"❌ Ошибка при создании векторной базы: {e}")
            return None
        
        return self.vectorstore
    
    def load_existing(self):
        """Загружает существующую векторную базу"""
        try:
            if os.path.exists(self.persist_directory):
                print(f"📂 Загружаю векторную базу из {self.persist_directory}")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                count = self.vectorstore._collection.count()
                print(f"✅ Загружено {count} фрагментов")
                return self.vectorstore
            else:
                print(f"⚠️  Папка {self.persist_directory} не найдена")
                return None
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
    
    def search(self, query, k=3):
        """Поиск по векторной базе"""
        if not self.vectorstore:
            print("⚠️  Сначала создайте или загрузите векторную базу")
            return []
        
        print(f"\n🔍 Поиск: '{query}'")
        print("-" * 60)
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                print("🤷 Не найдено результатов")
                return []
            
            print(f"📑 Найдено {len(results)} результатов:")
            
            for i, doc in enumerate(results):
                source = doc.metadata.get('source', 'Неизвестно')
                content_preview = doc.page_content[:200].replace('\n', ' ')
                
                print(f"\n[{i+1}] 📄 Источник: {source}")
                print(f"    📝 Фрагмент: {content_preview}...")
                print(f"    📊 Длина: {len(doc.page_content)} символов")
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_stats(self):
        """Статистика векторной базы"""
        if not self.vectorstore:
            return {"status": "not_loaded"}
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "loaded",
                "total_chunks": count,
                "persist_directory": self.persist_directory
            }
        except:
            return {"status": "error"}