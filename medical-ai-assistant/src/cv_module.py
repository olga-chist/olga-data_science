"""
CV модуль медицинского ассистента
Включает:
1. Классификация опухолей мозга (MRI) - ResNet
2. Сегментация лёгких (X-Ray) - U-Net
3. Детекция камней в почках (CT) - YOLOv8
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
from pathlib import Path

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (сжатие)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck (самое узкое место)
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (расширение)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Выходной слой
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder с skip-connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))

class CVMedicalAssistant:
    def __init__(self):
        print("CV модуль инициализирован")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain_model = None
        self.lung_model = None
        self.kidney_model = None
        
         # Трансформы для МРТ
        self.mri_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Классы для МРТ
        self.brain_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    def load_brain_model(self, model_path, class_info_path=None):
        """Загрузка ResNet модели для классификации МРТ мозга"""
        try:
            # Загружаем модель ResNet
            from torch.serialization import safe_globals
            with safe_globals([models.resnet.ResNet]):
                self.brain_model = torch.load(model_path, 
                                            map_location=self.device, 
                                            weights_only=False)
            self.brain_model.eval()
            print("✅ Модель классификации МРТ загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели МРТ: {e}")
            return False

    def load_lung_model(self, model_path):
        """Загрузка U-Net модели для сегментации лёгких (ТОЛЬКО ВЕСА)"""
        try:
          print(f"🔄 Загружаю веса модели лёгких...")
        
          # 1. Создаём архитектуру UNet (наш класс)
          self.lung_model = UNet(in_channels=1, out_channels=1)
        
          # 2. Загружаем ТОЛЬКО веса (state_dict)
          weights_path = model_path.replace(".pth", "_weights.pth")
          print(f"   Веса из: {weights_path}")
          state_dict = torch.load(weights_path, map_location=self.device)
        
          # 3. Загружаем веса в нашу архитектуру
          self.lung_model.load_state_dict(state_dict)
          self.lung_model.to(self.device)
          self.lung_model.eval()
        
          print("✅ Модель сегментации лёгких загружена (веса + наша архитектура)")
          print(f"Модель лёгких тип: {type(self.lung_model)}")
          print(f"Модель лёгких устройство: {next(self.lung_model.parameters()).device}")
          print(f"Параметры модели: {sum(p.numel() for p in self.lung_model.parameters())}")
          return True

        except Exception as e:
          print(f"❌ Ошибка загрузки модели лёгких: {e}")
          return False

    def load_kidney_model(self, model_path):
        """Загружаем YOLO модель для детекции камней"""
        try:
            from ultralytics import YOLO
            self.kidney_model = YOLO(model_path)
            print("✅ Модель детекции камней загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
    
    def classify_brain_mri(self, image_path):
        """Классификация опухолей мозга на МРТ"""
        if self.brain_model is None:
            return "Сначала загрузите модель МРТ", None, None
        
        try:
            # Загрузка и преобразование изображения
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.mri_transform(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.brain_model(image_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_prob, pred_class = torch.max(probs, 1)
            
            # Результат
            diagnosis = self.brain_classes[pred_class.item()]
            confidence = pred_prob.item()
            
            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Исходное изображение
            axes[0].imshow(image)
            axes[0].set_title("МРТ мозга")
            axes[0].axis('off')
            
            # График вероятностей
            y_pos = np.arange(len(self.brain_classes))
            axes[1].barh(y_pos, probs.cpu().numpy().flatten())
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(self.brain_classes)
            axes[1].set_xlabel("Вероятность")
            axes[1].set_title(f"Диагноз: {diagnosis}\nУверенность: {confidence:.2%}")
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            
            return fig, diagnosis, confidence
            
        except Exception as e:
            return f"Ошибка при обработке МРТ: {e}", None, None

    def segment_lungs(self, image_path):
        """Сегментация лёгких на рентгене"""
        if self.lung_model is None:
        # ВОТ ЭТО ИСПРАВИТЬ - ВОЗВРАЩАТЬ FIGURE, А НЕ СТРОКУ
          fig, ax = plt.subplots(1, 1, figsize=(8, 8))
          ax.text(0.5, 0.5, "Модель лёгких не загружена", 
                ha='center', va='center', fontsize=14)
          ax.axis('off')
          return fig, "Ошибка: модель не загружена"  # ← fig и текст
        
        try:
            # Загрузка изображения (упрощенно)
            import torchvision.transforms.functional as TF
            from PIL import Image
            
            image = Image.open(image_path).convert('L')  # В оттенки серого
            image = image.resize((256, 256))
            image_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                pred = self.lung_model(image_tensor)
                pred_binary = (pred > 0.5).float()
            
            if pred_binary is not None:
              # Считаем площадь в пикселях
              area_pixels = torch.sum(pred_binary > 0.5).item()
        
              # Добавляем в текст
              status_text = f"Сегментация лёгких выполнена. Площадь: {area_pixels} px²"
            else:
              status_text = "Сегментация лёгких выполнена"
              
            # Визуализация
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            img_display = image_tensor[0][0].cpu().numpy()
            pred_display = pred[0][0].cpu().numpy()
            binary_display = pred_binary[0][0].cpu().numpy()
            
            axes[0].imshow(img_display, cmap='gray')
            axes[0].set_title("Рентген лёгких")
            axes[0].axis('off')
            
            axes[1].imshow(pred_display, cmap='gray')
            axes[1].set_title("Предсказание")
            axes[1].axis('off')
            
            axes[2].imshow(binary_display, cmap='gray')
            axes[2].set_title("Сегментация")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            return fig, status_text
            
        except Exception as e:
            # ДАЖЕ ПРИ ОШИБКЕ ВОЗВРАЩАЕМ FIGURE
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.text(0.5, 0.5, f"Ошибка сегментации: {str(e)[:100]}", 
                ha='center', va='center', fontsize=12, wrap=True)
            ax.axis('off')
            return fig, f"Ошибка: {e}"

    def detect_kidney_stones(self, image_path, conf_threshold=0.4):
        """Детекция камней в почках на КТ"""
        if self.kidney_model is None:
            return "Сначала загрузите модель камней", 0
        
        try:
            # Предсказание
            results = self.kidney_model.predict(image_path, conf=conf_threshold)
            
            # Загрузка изображения
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_with_boxes = img.copy()
            
            # Обработка результатов
            stone_count = 0
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                stone_count = len(boxes)
                
                # Рисуем рамки
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Stone {conf:.2f}"
                    cv2.putText(img_with_boxes, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(img)
            axes[0].set_title("Исходное КТ")
            axes[0].axis('off')
            
            axes[1].imshow(img_with_boxes)
            axes[1].set_title(f"Обнаружено камней: {stone_count}")
            axes[1].axis('off')
            
            plt.tight_layout()
            
            return fig, stone_count
            
        except Exception as e:
            return f"Ошибка при детекции: {e}", 0

# Создаем экземпляр для импорта
if __name__ == "__main__":
    assistant = CVMedicalAssistant()
    print("✅ CV модуль готов. Используйте assistant.load_*_model() для загрузки моделей.")
