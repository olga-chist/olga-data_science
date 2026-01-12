"""
Мультимодальный медицинский ассистент
Объединяет CV анализ и RAG-бота
"""
import sys
import gradio as gr
import cv2
import tempfile
from pathlib import Path

# Импортируем наши модули
sys.path.append(str(Path(__file__).parent))
from src.cv_module import CVMedicalAssistant
from src.pdf_loader import MedicalPDFLoader
from src.vector_store import MedicalVectorStore
from src.medical_bot import MedicalRAGBot

class MultimodalMedicalAssistant:
    def __init__(self):
        print("🚀 Инициализация мультимодального ассистента...")
        
        # 1. Инициализируем CV модуль
        self.cv_assistant = CVMedicalAssistant()
        
        # 2. Загружаем CV модели
        print("🔄 Загрузка CV моделей...")
        
        # Модель детекции камней
        kidney_path = "/content/drive/MyDrive/medical-project/cv_models/kidney_stones/best.pt"
        self.cv_assistant.load_kidney_model(kidney_path)
        
        # Модель МРТ мозга
        brain_path = "/content/drive/MyDrive/medical-project/cv_models/brain_mri/brain_mri_classifier.pth"
        if Path(brain_path).exists():
            self.cv_assistant.load_brain_model(brain_path)
            
        else:
            print("⚠️ Файл модели МРТ не найден:", brain_path)
        
        # Модель сегментации лёгких
        lung_path = "/content/drive/MyDrive/medical-project/cv_models/lung_xray/lung_segmentation_unet.pth"
        if Path(lung_path).exists():
            self.cv_assistant.load_lung_model(lung_path)
            
        else:
            print("⚠️ Файл модели лёгких не найден:", lung_path)
        
        # 3. Инициализируем RAG-бота
        print("📚 Загрузка RAG-системы...")
        self.load_rag_bot()
        
        print("✅ Ассистент готов!")
    
    def load_rag_bot(self):
        """Загружаем RAG-бота с векторной БД"""
        try:
            # Загрузка документов и создание векторной БД
            loader = MedicalPDFLoader()
            documents = loader.load_all_documents()
            
            vector_store = MedicalVectorStore()
            if not vector_store.load_existing():
                vector_store.create_from_documents(documents)
            
            # Создаём RAG-бота
            model_path = "/content/drive/MyDrive/medical-project/models/model-q4_K.gguf"
            self.rag_bot = MedicalRAGBot(vector_store, model_path=model_path)
            
            print("✅ RAG-бот загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки RAG-бота: {e}")
            self.rag_bot = None
    
    def analyze_image(self, image, analysis_type):
        """
        Анализ медицинского изображения
        Возвращает: (диагноз_текст, визуализация_график)
        """
        try:
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                img_path = tmp.name
            
            # Выбираем тип анализа
            if analysis_type == "МРТ мозга":
                fig, diagnosis, confidence = self.cv_assistant.classify_brain_mri(img_path)
                diagnosis_text = f"Диагноз: {diagnosis} (уверенность: {confidence:.2%})"
                
            elif analysis_type == "Рентген лёгких":
                fig, status = self.cv_assistant.segment_lungs(img_path)
                diagnosis_text = f"Результат: {status}"
                
            elif analysis_type == "КТ почек":
                fig, stone_count = self.cv_assistant.detect_kidney_stones(img_path)
                diagnosis_text = f"Обнаружено камней: {stone_count}"
            
            else:
                return "Неизвестный тип анализа", None
            
            return diagnosis_text, fig
            
        except Exception as e:
            return f"Ошибка анализа: {e}", None
    
    def explain_diagnosis(self, diagnosis_text):
      """
      Объяснение диагноза через RAG-бота
      """
      # ПРОВЕРКА: диагноз есть?
      if not diagnosis_text or diagnosis_text == "":
        return "⚠️ Сначала получите диагноз, нажав 'Анализировать изображение'"
    
      if self.rag_bot is None:
        return "RAG-бот не загружен. Загрузите PDF документы."
    
      try:
        # Формируем запрос для бота на основе диагноза
        prompt = f"""
        Объясни пациенту следующий медицинский диагноз простым языком:
        {diagnosis_text}
        
        Объясни:
        1. Что это значит?
        2. Насколько это серьёзно?
        3. Какие следующие шаги рекомендуются?
        4. Какие обследования нужны?
        
        Ответь на русском, будто объясняешь пациенту.
        """
        
        response = self.rag_bot.answer_question(prompt, use_llm=True)
        
        answer = response.get("answer", "Нет ответа")
        sources = response.get("sources", [])
        
        # Форматируем ответ с источниками
        formatted_answer = f"{answer}\n\n📚 Источники информации:\n"
        for i, source in enumerate(sources[:3], 1):
            formatted_answer += f"{i}. {source.get('source', 'Неизвестный источник')}\n"
        
        return formatted_answer
        
      except Exception as e:
        return f"Ошибка генерации объяснения: {e}"
    
    def ask_question(self, question):
        """
        Обычный вопрос к RAG-боту
        """
        if self.rag_bot is None:
            return "RAG-бот не загружен."
        
        try:
            response = self.rag_bot.answer_question(question, use_llm=True)
            answer = response.get("answer", "Нет ответа")
            
            # Добавляем источники
            if response.get("sources"):
                answer += "\n\n📚 Источники:\n"
                for i, source in enumerate(response["sources"][:2], 1):
                    answer += f"{i}. {source.get('source', '?')}\n"
            
            return answer
            
        except Exception as e:
            return f"Ошибка: {e}"

# Создаём экземпляр ассистента
assistant = MultimodalMedicalAssistant()

# Создаём Gradio интерфейс
with gr.Blocks(title="Медицинский AI Ассистент", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Медицинский AI Ассистент")
    gr.Markdown("### Анализ изображений + Консультация с медицинской базой знаний")
    
    with gr.Tab("📷 Анализ медицинских изображений"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Загрузите медицинское изображение")
                analysis_type = gr.Radio(
                    choices=["МРТ мозга", "Рентген лёгких", "КТ почек"],
                    label="Тип анализа",
                    value="КТ почек"
                )
                analyze_btn = gr.Button("🔍 Анализировать изображение", variant="primary")
                
            with gr.Column(scale=1):
                diagnosis_output = gr.Textbox(label="Результат анализа", lines=3)
                explain_btn = gr.Button("🤖 Объяснить диагноз", variant="secondary")
                explanation_output = gr.Textbox(label="Объяснение диагноза", lines=10)
        
        with gr.Row():
            plot_output = gr.Plot(label="Визуализация")
    
    with gr.Tab("💬 Медицинский чат"):
        with gr.Row():
            with gr.Column(scale=2):
                chat_input = gr.Textbox(
                    label="Задайте медицинский вопрос",
                    placeholder="Например: Что такое диабет? Какие симптомы инфаркта?"
                )
                ask_btn = gr.Button("📤 Задать вопрос", variant="primary")
            
            with gr.Column(scale=3):
                chat_output = gr.Textbox(label="Ответ ассистента", lines=15)
    
    # Обработчики событий
    analyze_btn.click(
        fn=assistant.analyze_image,
        inputs=[image_input, analysis_type],
        outputs=[diagnosis_output, plot_output]
    )
    
    explain_btn.click(
        fn=assistant.explain_diagnosis,
        inputs=[diagnosis_output],
        outputs=[explanation_output]
    )
    
    ask_btn.click(
        fn=assistant.ask_question,
        inputs=[chat_input],
        outputs=[chat_output]
    )

# Запуск
if __name__ == "__main__":
    demo.launch(debug=True, share=True)