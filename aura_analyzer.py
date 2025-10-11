import cv2
import numpy as np
from PIL import Image
import io
import time
import json
from typing import Dict, List, Optional
import torch
from transformers import pipeline

class AuraHealthAnalyzer:
    def __init__(self, use_huggingface: bool = True):
        self.use_huggingface = use_huggingface
        self.celebrity_database = self._load_celebrity_database()
        
        if use_huggingface:
            # Инициализация моделей для анализа ауры
            self.face_analyzer = pipeline("image-classification", 
                                        model="google/vit-base-patch16-224")
            self.emotion_analyzer = pipeline("image-classification", 
                                           model="dima806/facial_emotions_image_detection")
    
    def _load_celebrity_database(self) -> Dict:
        """Загружает базу данных знаменитостей"""
        # Загрузи из папки celebrities или создай базовую
        return {
            "celebrity_1": {
                "name": "Пример знаменитости",
                "aura_score": 85,
                "traits": ["энергичность", "уверенность"]
            }
        }
    
    def analyze_from_bytes(self, image_bytes: bytes, user_id: str = "default") -> Dict:
        """Анализирует ауру из байтов изображения"""
        try:
            start_time = time.time()
            
            # Конвертация bytes в изображение
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Анализ изображения
            aura_score, health_traits = self._analyze_image(image_np)
            
            # Сравнение с знаменитостями
            celebrity_match = self._find_celebrity_match(aura_score, health_traits)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "user_id": user_id,
                "scores": {
                    "aura_score": aura_score,
                    "energy_level": aura_score - 10,
                    "vitality": aura_score + 5
                },
                "health_traits": health_traits,
                "celebrity_match": celebrity_match,
                "processing_time": processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_3d_scan(self, front_bytes: bytes, left_bytes: bytes, right_bytes: bytes, user_id: str = "default") -> Dict:
        """Анализирует ауру по 3 ракурсам (фронт, лево, право)"""
        try:
            start_time = time.time()
            
            # Анализ каждого ракурса
            front_result = self._analyze_single_angle(front_bytes, "front")
            left_result = self._analyze_single_angle(left_bytes, "left") 
            right_result = self._analyze_single_angle(right_bytes, "right")
            
            # Объединение результатов
            combined_score = self._combine_3d_scores(front_result, left_result, right_result)
            health_traits = self._combine_3d_traits(front_result, left_result, right_result)
            
            celebrity_match = self._find_celebrity_match(combined_score, health_traits)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "user_id": user_id,
                "scores": {
                    "aura_score": combined_score,
                    "front_score": front_result["score"],
                    "left_score": left_result["score"], 
                    "right_score": right_result["score"],
                    "balance_score": self._calculate_balance(front_result, left_result, right_result)
                },
                "health_traits": health_traits,
                "celebrity_match": celebrity_match,
                "processing_time": processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scan_type": "3d"
            }
            
        except Exception as e:
            return {"error": f"3D analysis failed: {str(e)}"}
    
    def _analyze_single_angle(self, image_bytes: bytes, angle: str) -> Dict:
        """Анализирует один ракурс"""
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        score, traits = self._analyze_image(image_np)
        
        return {
            "angle": angle,
            "score": score,
            "traits": traits
        }
    
    def _analyze_image(self, image_np: np.ndarray) -> tuple:
        """Основной анализ изображения - ЗДЕСЬ БУДЕТ ТВОЯ МОДЕЛЬ"""
        # TODO: Вставить твою реальную модель анализа ауры
        
        # Временная заглушка - случайные значения для демо
        aura_score = np.random.randint(50, 95)
        
        sample_traits = ["энергичность", "гармония", "уравновешенность", "творчество"]
        health_traits = np.random.choice(sample_traits, 2, replace=False).tolist()
        
        return aura_score, health_traits
    
    def _combine_3d_scores(self, front: Dict, left: Dict, right: Dict) -> int:
        """Объединяет оценки 3 ракурсов"""
        return int((front["score"] + left["score"] + right["score"]) / 3)
    
    def _combine_3d_traits(self, front: Dict, left: Dict, right: Dict) -> List[str]:
        """Объединяет черты 3 ракурсов"""
        all_traits = set(front["traits"] + left["traits"] + right["traits"])
        return list(all_traits)[:4]  # Максимум 4 черты
    
    def _calculate_balance(self, front: Dict, left: Dict, right: Dict) -> int:
        """Рассчитывает баланс между ракурсами"""
        scores = [front["score"], left["score"], right["score"]]
        balance = 100 - (np.std(scores) * 2)  # Чем меньше отклонение, тем выше баланс
        return max(0, min(100, int(balance)))
    
    def _find_celebrity_match(self, aura_score: int, traits: List[str]) -> Dict:
        """Находит совпадение со знаменитостью"""
        # Упрощенная логика сопоставления
        best_match = None
        min_diff = float('inf')
        
        for celeb_id, celeb_data in self.celebrity_database.items():
            diff = abs(aura_score - celeb_data["aura_score"])
            if diff < min_diff:
                min_diff = diff
                best_match = {
                    "name": celeb_data["name"],
                    "match_percentage": max(0, 100 - diff),
                    "shared_traits": [t for t in traits if t in celeb_data.get("traits", [])]
                }
        
        return best_match or {
            "name": "Не найдено",
            "match_percentage": 0,
            "shared_traits": []
        }
    
    def get_trends_data(self, user_id: str) -> Dict:
        """Возвращает историю анализов (заглушка)"""
        return {
            "user_id": user_id,
            "history": [],
            "trend": "stable"
        }
    
    def get_radar_chart_data(self, scan_id: str) -> Dict:
        """Возвращает данные для радар-диаграммы (заглушка)"""
        return {
            "scan_id": scan_id,
            "categories": ["Энергия", "Баланс", "Гармония", "Здоровье", "Творчество"],
            "scores": [75, 80, 65, 70, 85]
        }

# Для обратной совместимости
def analyze_3_images(front_bytes: bytes, left_bytes: bytes, right_bytes: bytes) -> Dict:
    """Упрощенная функция для анализа 3 изображений"""
    analyzer = AuraHealthAnalyzer(use_huggingface=True)
    return analyzer.analyze_3d_scan(front_bytes, left_bytes, right_bytes)