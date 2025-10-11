import cv2
import numpy as np
import json
from datetime import datetime
import mediapipe as mp
from transformers import pipeline
import torch
import hashlib
import time
import os
import glob

class AuraHealthAnalyzer:
    def __init__(self, use_huggingface=True):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        self.celebrity_database = self._initialize_celebrity_database()
        self.analysis_history = []
        self.use_huggingface = use_huggingface
        
        if use_huggingface:
            self._initialize_huggingface_models()

    def _initialize_huggingface_models(self):
        """Initialize Hugging Face models for emotion detection"""
        try:
            self.emotion_detector = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Hugging Face models loaded successfully")
        except Exception as e:
            print(f"❌ Could not load Hugging Face models: {e}")
            self.use_huggingface = False

    def _initialize_celebrity_database(self):
        """Загружает базу знаменитостей из папки celebrities"""
        celeb_folder = "celebrities"
        
        if not os.path.exists(celeb_folder):
            print("❌ Папка celebrities не найдена! Создай папку с фото знаменитостей")
            return {}
        
        image_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
            image_files.extend(glob.glob(os.path.join(celeb_folder, ext)))
        
        print(f"📁 Найдено {len(image_files)} фото знаменитостей")
        
        celebrity_database = {}
        
        for image_path in image_files:
            try:
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    continue
                
                filename = os.path.basename(image_path)
                celeb_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                
                print(f"🔍 Анализируем: {celeb_name}")
                
                celeb_metrics = self._analyze_celebrity_photo(image_path)
                if celeb_metrics:
                    celebrity_database[celeb_name] = {
                        "name": celeb_name,
                        "image_path": image_path,
                        "metrics": celeb_metrics
                    }
                    print(f"✅ Добавлена: {celeb_name}")
                else:
                    print(f"❌ Не удалось проанализировать: {celeb_name}")
                    
            except Exception as e:
                print(f"❌ Ошибка {os.path.basename(image_path)}: {str(e)}")
        
        print(f"✅ Загружено {len(celebrity_database)} знаменитостей")
        return celebrity_database

    def _analyze_celebrity_photo(self, image_path):
        """Анализирует фото знаменитости и возвращает метрики"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Базовая обработка
            img = cv2.resize(img, (500, 500))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Детекция лица
            results = self.face_detector.process(rgb_img)
            if not results.detections:
                return None
            
            # Обрезаем лицо
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = img.shape[:2]
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)
            
            margin = 0.1
            x = max(0, int(x - face_w * margin))
            y = max(0, int(y - face_h * margin))
            face_w = min(w - x, int(face_w * (1 + 2 * margin)))
            face_h = min(h - y, int(face_h * (1 + 2 * margin)))
            
            face_img = img[y:y+face_h, x:x+face_w]
            
            if face_img.size == 0:
                return None
            
            # Анализируем качество кожи
            skin_quality = self._analyze_skin_quality(face_img)
            
            # Анализируем свежесть глаз
            eye_freshness = self._analyze_eye_freshness(face_img)
            
            return {
                "skin_evenness": skin_quality["evenness"],
                "skin_smoothness": skin_quality["smoothness"], 
                "skin_cleanliness": skin_quality["cleanliness"],
                "eye_freshness": eye_freshness,
                "energy_level": 0.8 + (eye_freshness * 0.2)
            }
            
        except Exception as e:
            print(f"❌ Ошибка анализа {image_path}: {str(e)}")
            return None

    def analyze_3d_scan_from_bytes(self, front_bytes: bytes, left_bytes: bytes, right_bytes: bytes, user_id: str = "default"):
        """Основной метод анализа 3 изображений для бэкенда"""
        start_time = time.time()
        
        try:
            # Анализируем каждое изображение из bytes
            front_data = self._analyze_frontal_from_bytes(front_bytes)
            left_data = self._analyze_profile_from_bytes(left_bytes)
            right_data = self._analyze_profile_from_bytes(right_bytes)
            
            # Рассчитываем скоры
            scores = self._calculate_scores(front_data, left_data, right_data)
            metrics = self._compile_metrics(front_data)
            
            # Генерируем рекомендации
            recommendations = self._generate_recommendations(scores, metrics)
            celebrity_match = self.get_celebrity_match({"metrics": metrics})
            
            processing_time = round(time.time() - start_time, 2)
            
            result = {
                'scan_id': self._generate_scan_id(),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'scores': scores,
                'metrics': metrics,
                'recommendations': recommendations,
                'celebrity_match': celebrity_match,
                'processing_time': processing_time
            }
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

    def _analyze_frontal_from_bytes(self, image_bytes: bytes):
        """Анализ фронтального фото из bytes"""
        processed_img = self._preprocess_image_from_bytes(image_bytes)
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        mesh_results = self.face_mesh.process(rgb_img)
        emotion_analysis = self._analyze_emotions(rgb_img) if self.use_huggingface else {}
        
        return {
            'energy': self._analyze_energy(processed_img, mesh_results),
            'skin': self._analyze_skin_quality(processed_img),
            'stress': self._analyze_stress(processed_img, mesh_results, emotion_analysis)
        }

    def _analyze_profile_from_bytes(self, image_bytes: bytes):
        """Анализ профильного фото из bytes"""
        try:
            processed_img = self._preprocess_image_from_bytes(image_bytes)
            return {
                'puffiness': 0.3,
                'symmetry': 0.8
            }
        except:
            return {}

    def _preprocess_image_from_bytes(self, image_bytes: bytes):
        """Препроцессинг изображения из bytes"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes")
        
        img = self._normalize_lighting(img)
        img = self._white_balance(img)
        img = self._denoise_image(img)
        face_img = self._detect_face_modern(img)
        
        return face_img

    def _normalize_lighting(self, img):
        """Нормализация освещения"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_normalized = clahe.apply(l)
        lab_normalized = cv2.merge([l_normalized, a, b])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    def _white_balance(self, img):
        """Коррекция баланса белого"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _denoise_image(self, img):
        """Удаление шумов"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def _detect_face_modern(self, img):
        """Детекция лица с MediaPipe"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_img)
        
        if not results.detections:
            raise ValueError("No face detected")
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = img.shape[:2]
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        face_w = int(bbox.width * w)
        face_h = int(bbox.height * h)
        
        margin = 0.1
        x = max(0, int(x - face_w * margin))
        y = max(0, int(y - face_h * margin))
        face_w = min(w - x, int(face_w * (1 + 2 * margin)))
        face_h = min(h - y, int(face_h * (1 + 2 * margin)))
        
        return img[y:y+face_h, x:x+face_w]

    def _analyze_energy(self, face_img, mesh_results):
        """Анализ энергии и усталости"""
        dark_circle_score = self._analyze_dark_circles(face_img)
        eye_freshness = self._analyze_eye_freshness(face_img)
        
        energy_score = 100 - (dark_circle_score * 40 + (1 - eye_freshness) * 40)
        
        return {
            'score': max(0, min(energy_score, 100)),
            'dark_circles': dark_circle_score,
            'eye_freshness': eye_freshness
        }

    def _analyze_dark_circles(self, face_img):
        """Анализ темных кругов под глазами"""
        eye_regions = self._extract_eye_regions(face_img)
        
        if len(eye_regions) < 2:
            return 0.5
        
        dark_scores = []
        for eye_region in eye_regions:
            h, w = eye_region.shape[:2]
            under_eye = eye_region[int(h*0.7):h, :]
            
            if under_eye.size == 0:
                continue
            
            hsv = cv2.cvtColor(under_eye, cv2.COLOR_RGB2HSV)
            
            lower_blue = np.array([100, 40, 20])
            upper_blue = np.array([140, 255, 150])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            lower_brown = np.array([0, 40, 20])
            upper_brown = np.array([20, 255, 150])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            combined_mask = cv2.bitwise_or(blue_mask, brown_mask)
            dark_ratio = np.sum(combined_mask > 0) / combined_mask.size
            
            dark_scores.append(min(dark_ratio * 2, 1.0))
        
        return np.mean(dark_scores) if dark_scores else 0.0

    def _extract_eye_regions(self, face_img):
        """Извлечение областей глаз по landmarks"""
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_face)
        
        if not mesh_results.multi_face_landmarks:
            return []
        
        eye_regions = []
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for eye_indices in [left_eye_indices, right_eye_indices]:
            points = []
            for idx in eye_indices:
                x = int(landmarks[idx].x * face_img.shape[1])
                y = int(landmarks[idx].y * face_img.shape[0])
                points.append([x, y])
            
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(face_img.shape[1], x_max + padding)
                y_max = min(face_img.shape[0], y_max + padding)
                
                eye_region = face_img[y_min:y_max, x_min:x_max]
                if eye_region.size > 0:
                    eye_regions.append(eye_region)
        
        return eye_regions

    def _analyze_eye_freshness(self, face_img):
        """Анализ свежести глаз"""
        eye_regions = self._extract_eye_regions(face_img)
        
        if len(eye_regions) < 2:
            return 0.5
        
        freshness_scores = []
        for eye_region in eye_regions:
            h, w = eye_region.shape[:2]
            eye_white = eye_region[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            
            if eye_white.size == 0:
                continue
            
            brightness = np.mean(cv2.cvtColor(eye_white, cv2.COLOR_RGB2GRAY))
            freshness_scores.append(brightness / 255.0)
        
        return np.mean(freshness_scores) if freshness_scores else 0.5

    def _analyze_skin_quality(self, face_img):
        """Анализ качества кожи"""
        face_resized = cv2.resize(face_img, (200, 200))
        
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_RGB2HSV)
        color_evenness = max(0, 1 - np.std(hsv[:,:,0]) / 30)
        
        gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        smoothness = max(0, 1 - cv2.Laplacian(gray, cv2.CV_64F).var() / 500)
        
        cleanliness = self._analyze_skin_cleanliness(face_resized)
        
        skin_score = (color_evenness * 0.4 + smoothness * 0.4 + cleanliness * 0.2) * 100
        
        return {
            'score': max(0, min(skin_score, 100)),
            'evenness': color_evenness,
            'smoothness': smoothness,
            'cleanliness': cleanliness
        }

    def _analyze_skin_cleanliness(self, face_img):
        """Анализ чистоты кожи"""
        hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        return max(0, 1 - red_ratio * 3)

    def _analyze_stress(self, face_img, mesh_results, emotion_analysis):
        """Анализ стресса"""
        brow_tension = 0.3
        mouth_tension = 0.4
        
        base_stress_score = 100 - (brow_tension * 40 + mouth_tension * 35)
        
        if emotion_analysis:
            emotion_stress = emotion_analysis.get('computed_stress', base_stress_score)
            combined_stress = (base_stress_score * 0.7 + emotion_stress * 0.3)
            base_stress_score = max(0, min(combined_stress, 100))
        
        return {
            'score': base_stress_score,
            'brow_tension': brow_tension,
            'mouth_tension': mouth_tension
        }

    def _analyze_emotions(self, img_rgb):
        """Анализ эмоций через Hugging Face"""
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            emotion_results = self.emotion_detector(pil_img)
            
            stress_indicators = ['sad', 'angry', 'fear', 'disgust']
            stress_score = 0
            for result in emotion_results:
                if result['label'].lower() in stress_indicators:
                    stress_score += result['score']
            
            return {
                'emotions': emotion_results,
                'computed_stress': min(stress_score * 100, 100),
                'dominant_emotion': emotion_results[0]['label'] if emotion_results else 'neutral'
            }
        except Exception as e:
            print(f"Emotion analysis failed: {e}")
            return {}

    def _calculate_scores(self, front_data, left_data, right_data):
        """Расчет итоговых скоров"""
        energy_score = front_data['energy']['score']
        skin_score = front_data['skin']['score']
        stress_score = front_data['stress']['score']
        
        aura_score = (energy_score * 0.4 + skin_score * 0.35 + stress_score * 0.25)
        
        return {
            'aura_score': round(aura_score, 1),
            'energy_score': round(energy_score, 1),
            'skin_score': round(skin_score, 1),
            'stress_score': round(stress_score, 1)
        }

    def _compile_metrics(self, front_data):
        """Компиляция всех метрик"""
        return {
            'dark_circles': front_data['energy']['dark_circles'],
            'eye_freshness': front_data['energy']['eye_freshness'],
            'skin_smoothness': front_data['skin']['smoothness'],
            'skin_evenness': front_data['skin']['evenness'],
            'skin_cleanliness': front_data['skin']['cleanliness']
        }

    def _generate_recommendations(self, scores, metrics):
        """Генерация рекомендаций"""
        recommendations = []
        
        if metrics['dark_circles'] > 0.1:
            confidence = min(metrics['dark_circles'] * 5, 1.0)
            priority = 'high' if metrics['dark_circles'] > 0.2 else 'medium'
            recommendations.append({
                'category': 'energy',
                'priority': priority,
                'message': 'Темные круги под глазами. Рекомендуется: сон 7-8 часов, холодные компрессы',
                'action': 'improve_sleep',
                'confidence': round(confidence, 2)
            })
        
        if scores['skin_score'] < 60:
            recommendations.append({
                'category': 'skin',
                'priority': 'high' if scores['skin_score'] < 40 else 'medium',
                'message': 'Коже требуется уход. Используйте увлажняющие средства',
                'action': 'skin_hydration',
                'confidence': round(1 - (scores['skin_score'] / 100), 2)
            })
        
        if scores['stress_score'] < 70:
            recommendations.append({
                'category': 'stress',
                'priority': 'medium',
                'message': 'Повышенный уровень стресса. Попробуйте дыхательные техники',
                'action': 'stress_management',
                'confidence': round(1 - (scores['stress_score'] / 100), 2)
            })
        
        recommendations.sort(key=lambda x: (x['priority'] == 'high', x['confidence']), reverse=True)
        return recommendations

    def get_celebrity_match(self, user_scan):
        """Поиск похожей знаменитости"""
        if not self.celebrity_database:
            return {"error": "База знаменитостей пуста"}
        
        best_match = None
        best_similarity = 0
        
        for celeb_name, celeb_data in self.celebrity_database.items():
            similarity = self._calculate_similarity(user_scan, celeb_data)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = celeb_data
        
        if best_match and best_similarity > 0.4:
            return {
                "celebrity": best_match["name"],
                "similarity_percent": round(best_similarity * 100, 1),
                "verdict": f"Вы похожи на {best_match['name']} на {round(best_similarity * 100, 1)}%!"
            }
        else:
            return {
                "celebrity": None,
                "verdict": "Уникальный профиль! Вы особенные! ✨"
            }

    def _calculate_similarity(self, user_scan, celeb_data):
        """Расчет схожести по метрикам"""
        user_metrics = user_scan["metrics"]
        celeb_metrics = celeb_data["metrics"]
        
        total_similarity = 0
        metric_count = 0
        
        for metric, celeb_value in celeb_metrics.items():
            user_value = user_metrics.get(metric, 0)
            similarity = 1 - abs(user_value - celeb_value)
            total_similarity += max(0, similarity)
            metric_count += 1
        
        return total_similarity / metric_count if metric_count > 0 else 0

    def _generate_scan_id(self):
        timestamp = str(time.time())
        return f"scan_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"

# Функция для бэкенда
def analyze_3_images(front_bytes: bytes, left_bytes: bytes, right_bytes: bytes) -> dict:
    """Основная функция для анализа 3 изображений"""
    analyzer = AuraHealthAnalyzer(use_huggingface=True)
    return analyzer.analyze_3d_scan_from_bytes(front_bytes, left_bytes, right_bytes)