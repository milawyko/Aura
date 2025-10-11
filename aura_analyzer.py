import cv2
import numpy as np
import json
from datetime import datetime
import mediapipe as mp
from transformers import pipeline
import torch
import hashlib
import time

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
        """Celebrity database for comparison"""
        return {
            "Криштиану Роналду": {
                "name": "Криштиану Роналду",
                "metrics": {
                    "skin_evenness": 0.9,
                    "skin_smoothness": 0.85,
                    "skin_cleanliness": 0.95,
                    "eye_freshness": 0.95,
                    "energy_level": 0.98
                }
            },
            "Марго Робби": {
                "name": "Марго Робби", 
                "metrics": {
                    "skin_evenness": 0.95,
                    "skin_smoothness": 0.92,
                    "skin_cleanliness": 0.90,
                    "eye_freshness": 0.88,
                    "energy_level": 0.85
                }
            }
        }

    def analyze_3d_scan_from_bytes(self, front_bytes: bytes, left_bytes: bytes, right_bytes: bytes, user_id: str = "default"):
        """Main analysis method for 3 images (bytes) - ДЛЯ БЭКЕНДА"""
        start_time = time.time()
        
        try:
            # Analyze each image from bytes
            front_data = self._analyze_frontal_from_bytes(front_bytes)
            left_data = self._analyze_profile_from_bytes(left_bytes)
            right_data = self._analyze_profile_from_bytes(right_bytes)
            
            # Calculate scores
            scores = self._calculate_scores(front_data, left_data, right_data)
            metrics = self._compile_metrics(front_data)
            
            # Generate recommendations
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
        """Analyze frontal photo from bytes"""
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
        """Analyze profile photo from bytes"""
        try:
            processed_img = self._preprocess_image_from_bytes(image_bytes)
            return {
                'puffiness': 0.3,
                'symmetry': 0.8
            }
        except:
            return {}

    def _preprocess_image_from_bytes(self, image_bytes: bytes):
        """Preprocess image from bytes"""
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
        """Lighting normalization using CLAHE"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_normalized = clahe.apply(l)
        lab_normalized = cv2.merge([l_normalized, a, b])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    def _white_balance(self, img):
        """White balance correction"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _denoise_image(self, img):
        """Image denoising"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def _detect_face_modern(self, img):
        """Modern face detection with MediaPipe"""
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

    # Остальные методы анализа (_analyze_energy, _analyze_skin_quality, и т.д.)
    # остаются без изменений из твоего кода...

    def _analyze_energy(self, face_img, mesh_results):
        """Analyze energy and fatigue"""
        dark_circle_score = self._analyze_dark_circles(face_img)
        eye_freshness = self._analyze_eye_freshness(face_img)
        muscle_tone = self._analyze_muscle_tone(mesh_results)
        
        energy_score = 100 - (dark_circle_score * 40 + (1 - eye_freshness) * 40 + (1 - muscle_tone) * 20)
        
        return {
            'score': max(0, min(energy_score, 100)),
            'dark_circles': dark_circle_score,
            'eye_freshness': eye_freshness,
            'muscle_tone': muscle_tone
        }

    def _analyze_skin_quality(self, face_img):
        """Analyze skin quality"""
        face_resized = cv2.resize(face_img, (200, 200))
        
        # Color evenness
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_RGB2HSV)
        color_evenness = max(0, 1 - np.std(hsv[:,:,0]) / 30)
        
        # Texture smoothness
        gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        smoothness = max(0, 1 - cv2.Laplacian(gray, cv2.CV_64F).var() / 500)
        
        # Skin cleanliness
        cleanliness = self._analyze_skin_cleanliness(face_resized)
        
        skin_score = (color_evenness * 0.4 + smoothness * 0.4 + cleanliness * 0.2) * 100
        
        return {
            'score': max(0, min(skin_score, 100)),
            'evenness': color_evenness,
            'smoothness': smoothness,
            'cleanliness': cleanliness
        }

    def _calculate_scores(self, front_data, left_data, right_data):
        """Calculate final scores"""
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
        """Compile all metrics"""
        return {
            'dark_circles': front_data['energy']['dark_circles'],
            'eye_freshness': front_data['energy']['eye_freshness'],
            'skin_smoothness': front_data['skin']['smoothness'],
            'skin_evenness': front_data['skin']['evenness'],
            'skin_cleanliness': front_data['skin']['cleanliness']
        }

    def _generate_recommendations(self, scores, metrics):
        """Generate intelligent recommendations"""
        recommendations = []
        
        if metrics['dark_circles'] > 0.1:
            confidence = min(metrics['dark_circles'] * 5, 1.0)
            priority = 'high' if metrics['dark_circles'] > 0.2 else 'medium'
            recommendations.append({
                'category': 'energy',
                'priority': priority,
                'message': 'Dark circles detected. Recommended: 7-8 hours sleep, cold compresses',
                'action': 'improve_sleep',
                'confidence': round(confidence, 2)
            })
        
        if scores['skin_score'] < 60:
            recommendations.append({
                'category': 'skin',
                'priority': 'high' if scores['skin_score'] < 40 else 'medium',
                'message': 'Skin requires care. Use moisturizing products',
                'action': 'skin_hydration',
                'confidence': round(1 - (scores['skin_score'] / 100), 2)
            })
        
        recommendations.sort(key=lambda x: (x['priority'] == 'high', x['confidence']), reverse=True)
        return recommendations

    def get_celebrity_match(self, user_scan):
        """Find celebrity match"""
        if not self.celebrity_database:
            return {"error": "Celebrity database empty"}
        
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
        """Calculate similarity metrics"""
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

# Simplified function for backend
def analyze_3_images(front_bytes: bytes, left_bytes: bytes, right_bytes: bytes) -> dict:
    """Main function for backend - analyzes 3 images"""
    analyzer = AuraHealthAnalyzer(use_huggingface=True)
    return analyzer.analyze_3d_scan_from_bytes(front_bytes, left_bytes, right_bytes)