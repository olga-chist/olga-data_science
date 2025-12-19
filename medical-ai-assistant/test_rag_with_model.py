"""
Тестирование медицинского RAG-бота с BioMistral моделью
"""
import sys
sys.path.append('/content/drive/MyDrive/medical-project')

from src.pdf_loader import MedicalPDFLoader
from src.vector_store import MedicalVectorStore
from src.medical_bot import MedicalRAGBot

print("🚀 ТЕСТИРОВАНИЕ ПОЛНОГО RAG-БОТА С BIO-MISTRAL")
print("=" * 60)

# 1. Загружаем документы
print("\n📄 Этап 1: Загрузка документов...")
loader = MedicalPDFLoader()
documents = loader.load_all_documents()

if not documents:
    print("❌ Нет документов для обработки")
    exit()

# 2. Создаём/загружаем векторную базу
print("\n🧠 Этап 2: Векторная база...")
vector_store = MedicalVectorStore()

# Пробуем загрузить существующую
if not vector_store.load_existing():
    print("Создаю новую векторную базу...")
    vector_store.create_from_documents(documents)

# 3. Путь к модели BioMistral
model_path = "/content/models/BioMistral-7B.Q4_K_M.gguf"

# Проверяем, существует ли модель
import os
if not os.path.exists(model_path):
    print(f"\n❌ Модель не найдена по пути: {model_path}")
    print("Сначала скачайте модель командой выше")
    exit()

# 4. Создаём бота С моделью
print(f"\n🤖 Этап 3: Создание бота с BioMistral...")
bot = MedicalRAGBot(vector_store, model_path=model_path)

# 5. Тестовые вопросы (сначала без генерации для проверки)
print("\n🔍 Этап 4: Тест поиска (без генерации)...")
test_questions = [
    "Что такое диабет?",
    "Как предотвратить болезни сердца?",
    "Какое нормальное артериальное давление?"
]

for question in test_questions:
    print(f"\n{'='*50}")
    print(f"❓ Вопрос: {question}")
    
    # Сначала показываем найденные фрагменты
    search_response = bot.answer_question(question, use_llm=False)
    
    if search_response["sources"]:
        print(f"📄 Найдено фрагментов: {len(search_response['sources'])}")
        for i, source in enumerate(search_response["sources"]):
            print(f"  {i+1}. {source['source']}")
    else:
        print("🤷 Не найдено информации")
    
    print(f"⏱️ Время поиска: {search_response['search_time']:.2f} сек.")

# 6. Теперь с генерацией ответов (1 вопрос для экономии времени)
print("\n\n💭 Этап 5: Генерация ответа с BioMistral...")
print("=" * 60)
print("⚠️  Внимание: Генерация займет 1-3 минуты на вопрос")
print("=" * 60)

demo_question = "Что такое диабет и как его лечат?"
print(f"\n❓ Вопрос: {demo_question}")

try:
    print("⏳ Генерирую ответ...")
    full_response = bot.answer_question(demo_question, use_llm=True)
    
    print(f"\n✅ Ответ сгенерирован!")
    print(f"⏱️ Общее время: {full_response['search_time']:.2f} сек.")
    
    if full_response.get("answer"):
        print(f"\n🤖 Ответ BioMistral:")
        print("-" * 50)
        print(full_response["answer"])
        print("-" * 50)
    
    if full_response.get("sources"):
        print(f"\n📚 Источники информации:")
        for i, source in enumerate(full_response["sources"]):
            print(f"  {i+1}. {source['source']}")
    
except Exception as e:
    print(f"\n❌ Ошибка генерации: {e}")
    print("Возможные причины:")
    print("1. Не хватает памяти (Colab бесплатный)")
    print("2. Модель не скачалась полностью")
    print("3. Проблема с llama-cpp-python")

print("\n🎉 Тестирование завершено!")

# 7. Рекомендации
print("\n📋 Дальнейшие шаги:")
print("1. Если работает: добавь больше тестовых вопросов")
print("2. Если не хватает памяти: переключись на T4 GPU (Runtime → Change runtime type)")
print("3. Для проекта: добавь интерфейс (Gradio)")
print("4. Для портфолио: запиши видео-демо")