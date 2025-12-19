"""
Medical RAG Bot - отвечает на вопросы по медицинским документам
"""
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import time

class MedicalRAGBot:
    def __init__(self, vector_store, model_path=None):
        """
        Инициализация медицинского RAG-бота
        
        Args:
            vector_store: объект MedicalVectorStore с загруженными документами
            model_path: путь к модели BioMistral (если None, использует тестовый режим)
        """
        self.vector_store = vector_store
        self.qa_chain = None
        
        # Медицинский системный промпт
        self.prompt_template = """Ты - медицинский ассистент. Отвечай на вопросы, используя предоставленную информацию.
        Если информации недостаточно, скажи об этом. Отвечай точно и по делу.

        Контекст: {context}

        Вопрос: {question}

        Медицинский ответ:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Если передан путь к модели, загружаем её
        if model_path:
            self._load_model(model_path)
        else:
            print("⚠️ Режим без модели: будет показывать только найденные фрагменты")
    
    def _load_model(self, model_path):
        """Загрузка языковой модели (BioMistral)"""
        print(f"🧠 Загружаю модель из: {model_path}")
        start_time = time.time()
        
        try:
            # Настройки для BioMistral в Colab
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,           # Консервативные ответы (меньше креативности)
                max_tokens=512,            # Максимальная длина ответа
                top_p=0.95,                # Разнообразие ответов
                n_ctx=2048,                # Контекстное окно
                verbose=False              # Не выводить технические детали
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Модель загружена за {elapsed:.1f} сек.")
            
            # Создаём цепочку RAG
            self._create_qa_chain()
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("Продолжаю в режиме без LLM")
            self.llm = None
    
    def _create_qa_chain(self):
        """Создание цепочки вопрос-ответ с RAG"""
        if not self.vector_store.vectorstore:
            print("⚠️ Векторная база не загружена")
            return
        
        print("🔗 Создаю RAG-цепочку...")
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Простой способ объединения контекста
                retriever=self.vector_store.vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Берём 3 наиболее релевантных фрагмента
                ),
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True  # Чтобы видеть, откуда взята информация
            )
            print("✅ RAG-цепочка создана")
            
        except Exception as e:
            print(f"❌ Ошибка создания цепочки: {e}")
    
    def answer_question(self, question, use_llm=True):
        """
        Ответ на медицинский вопрос
        
        Args:
            question: строка с вопросом
            use_llm: использовать ли языковую модель (False - только поиск)
        
        Returns:
            dict с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print(f"❓ Вопрос: {question}")
        print(f"{'='*60}")
        
        # 1. Поиск релевантных фрагментов
        print("\n🔍 Ищу информацию в документах...")
        start_time = time.time()
        
        search_results = self.vector_store.search(question, k=3)
        
        if not search_results:
            return {
                "answer": "Не найдено информации по данному вопросу в документах.",
                "sources": [],
                "search_time": time.time() - start_time,
                "llm_used": False
            }
        
        # 2. Если нет модели или use_llm=False, показываем только найденное
        if not use_llm or not self.qa_chain:
            print("\n📄 Найденные фрагменты (без генерации ответа):")
            sources = []
            
            for doc in search_results:
                source_info = {
                    "source": doc.metadata.get('source', 'Неизвестно'),
                    "content": doc.page_content[:300] + "...",
                    "length": len(doc.page_content)
                }
                sources.append(source_info)
            
            return {
                "answer": None,
                "sources": sources,
                "search_time": time.time() - start_time,
                "llm_used": False
            }
        
        # 3. Генерация ответа с помощью LLM
        print("💭 Генерирую ответ...")
        
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "source": doc.metadata.get('source', 'Неизвестно'),
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in result["source_documents"]
                ],
                "search_time": time.time() - start_time,
                "llm_used": True
            }
            
            print(f"✅ Ответ сгенерирован")
            return response
            
        except Exception as e:
            print(f"❌ Ошибка генерации ответа: {e}")
            return {
                "answer": f"Ошибка при генерации ответа: {str(e)}",
                "sources": [],
                "search_time": time.time() - start_time,
                "llm_used": False
            }
    
    def test_basic_queries(self):
        """Тестовые медицинские вопросы"""
        test_queries = [
            "Что такое диабет?",
            "Как предотвратить болезни сердца?",
            "Каковы симптомы сердечного приступа?",
            "Что такое артериальное давление?"
        ]
        
        print("\n🧪 ТЕСТИРУЮ БОТА:")
        print("=" * 60)
        
        for query in test_queries:
            response = self.answer_question(query, use_llm=False)
            
            print(f"\nВопрос: {query}")
            if response["sources"]:
                print(f"Найдено источников: {len(response['sources'])}")
                for i, source in enumerate(response["sources"]):
                    print(f"  {i+1}. {source['source']}")
            else:
                print("Источники не найдены")
            
            print(f"Время поиска: {response['search_time']:.2f} сек.")
            print("-" * 40)