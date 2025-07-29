import cv2
import os
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import threading
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from skimage.feature import hog
import random
import glob
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import psutil

# Configuración de Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

# Crear directorio de uploads
os.makedirs('uploads', exist_ok=True)

# Variables globales para métricas del sistema
system_metrics = {
    'fps': 0,
    'memory': '0 MB',
    'cpu': '0%',
    'confidence': '0%',
    'faces_detected': 0,
    'objects_detected': 0,
    'last_update': time.time()
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class FaceDetectorLBP:
    def __init__(self):
        self.classifier = None
        self.svm_model = None
        self.is_trained = False
        self.training_progress = 0
        self.window_size = (64, 64)  # Tamaño de ventana para HOG
        
        # SIFT detector and matcher para detectar objetos específicos
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.reference_objects = {}  # Almacena características de objetos específicos subidos por el usuario
        self.detect_objects = False  # Flag para activar/desactivar detección de objetos
        self.anonymize_objects = True  # Flag para controlar si se anonimizan o solo se marcan los objetos
        
        # Métricas de rendimiento
        self.frame_times = []
        self.face_count = 0
        self.object_count = 0
        self.confidence_scores = []
        
    def extract_hog_features(self, image):
        """Extrae características HOG de una imagen"""
        # Redimensionar imagen
        image_resized = cv2.resize(image, self.window_size)
        
        # Convertir a escala de grises si es necesario
        if len(image_resized.shape) == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_resized
            
        # Extraer características HOG
        features = hog(image_gray, 
                      orientations=9, 
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), 
                      visualize=False)
        return features
    
    def load_dataset(self):
        """Carga y procesa el dataset completo"""
        print("Cargando dataset...")
        
        features = []
        labels = []
        
        # Cargar imágenes POSITIVAS (rostros)
        positive_path = "dataset/positive/person"
        positive_count = 0
        if os.path.exists(positive_path):
            for img_file in os.listdir(positive_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(positive_path, img_file)
                    try:
                        image = cv2.imread(img_path)
                        if image is not None:
                            # Extraer características HOG
                            hog_features = self.extract_hog_features(image)
                            features.append(hog_features)
                            labels.append(1)  # 1 = rostro
                            positive_count += 1
                            
                            if positive_count % 100 == 0:
                                print(f"Procesadas {positive_count} imágenes positivas...")
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
                        continue
        
        print(f"Total imágenes positivas procesadas: {positive_count}")
        
        # Cargar imágenes NEGATIVAS (no rostros)
        negative_path = "dataset/negative"
        negative_count = 0
        max_negatives = positive_count * 2  # Balancear dataset
        
        negative_files = []
        for category in os.listdir(negative_path):
            category_path = os.path.join(negative_path, category)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        negative_files.append(os.path.join(category_path, img_file))
        
        # Mezclar y limitar imágenes negativas
        random.shuffle(negative_files)
        negative_files = negative_files[:max_negatives]
        
        for img_path in negative_files:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    # Extraer características HOG
                    hog_features = self.extract_hog_features(image)
                    features.append(hog_features)
                    labels.append(0)  # 0 = no rostro
                    negative_count += 1
                    
                    if negative_count % 100 == 0:
                        print(f"Procesadas {negative_count} imágenes negativas...")
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue
        
        print(f"Total imágenes negativas procesadas: {negative_count}")
        print(f"Dataset total: {len(features)} imágenes")
        
        return np.array(features), np.array(labels)
    
    def train_classifier(self):
        """Entrena el clasificador SVM con características HOG REALMENTE"""
        print("=== INICIANDO ENTRENAMIENTO REAL ===")
        print("Esto puede tomar varios minutos...")
        
        try:
            # Paso 1: Cargar y procesar dataset
            self.training_progress = 10
            features, labels = self.load_dataset()
            
            if len(features) == 0:
                print("Error: No se pudieron cargar imágenes del dataset")
                return False
            
            self.training_progress = 30
            
            # Paso 2: Dividir en entrenamiento y prueba
            print("Dividiendo dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            self.training_progress = 50
            
            # Paso 3: Entrenar SVM
            print("Entrenando modelo SVM...")
            print(f"Datos de entrenamiento: {len(X_train)} muestras")
            print(f"Datos de prueba: {len(X_test)} muestras")
            
            self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            self.svm_model.fit(X_train, y_train)
            
            self.training_progress = 80
            
            # Paso 4: Evaluar modelo
            print("Evaluando modelo...")
            y_pred = self.svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Precisión del modelo: {accuracy:.2%}")
            
            self.training_progress = 90
            
            # Paso 5: Guardar modelo entrenado
            print("Guardando modelo...")
            os.makedirs("trained_model", exist_ok=True)
            with open("trained_model/face_detector_svm.pkl", "wb") as f:
                pickle.dump(self.svm_model, f)
            
            self.is_trained = True
            self.training_progress = 100
            
            print("=== ENTRENAMIENTO COMPLETADO ===")
            print(f"Modelo entrenado con {len(features)} imágenes")
            print(f"Precisión: {accuracy:.2%}")
            print("Modelo guardado en: trained_model/face_detector_svm.pkl")
            
            return True
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            return False
    
    def load_trained_model(self):
        """Carga un modelo previamente entrenado"""
        model_path = "trained_model/face_detector_svm.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.svm_model = pickle.load(f)
                self.is_trained = True
                self.training_progress = 100
                print("Modelo entrenado cargado exitosamente")
                return True
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
        return False
    
    def detect_faces(self, frame):
        """Detecta rostros usando híbrido: OpenCV rápido + modelo entrenado para validación"""
        start_time = time.time()
        
        # PASO 1: Usar detector rápido de OpenCV para encontrar candidatos
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        validated_faces = []
        
        # PASO 2: Si tenemos modelo entrenado, validar cada detección
        if self.is_trained and self.svm_model is not None:
            for (x, y, w, h) in faces:
                # Extraer región del rostro candidato
                face_region = frame[y:y+h, x:x+w]
                
                try:
                    # Extraer características HOG de la región
                    hog_features = self.extract_hog_features(face_region)
                    
                    # Validar con nuestro modelo entrenado
                    prediction = self.svm_model.predict([hog_features])[0]
                    probability = self.svm_model.predict_proba([hog_features])[0]
                    
                    # Si nuestro modelo también dice que es rostro
                    if prediction == 1 and probability[1] > 0.6:  # Umbral más permisivo
                        validated_faces.append((x, y, w, h, probability[1], "TRAINED"))
                    else:
                        # Usar detección básica si nuestro modelo no está seguro
                        validated_faces.append((x, y, w, h, 0.8, "BASIC"))
                except:
                    # Si hay error, usar detección básica
                    validated_faces.append((x, y, w, h, 0.8, "BASIC"))
        else:
            # Si no tenemos modelo entrenado, usar todas las detecciones básicas
            for (x, y, w, h) in faces:
                validated_faces.append((x, y, w, h, 0.8, "BASIC"))
        
        # PASO 3: Pixelar todos los rostros detectados
        for (x, y, w, h, conf, method) in validated_faces:
            # Extraer región del rostro
            face_region = frame[y:y+h, x:x+w]
            
            # Pixelar con diferentes intensidades según el método
            if method == "TRAINED":
                # Pixelación más fuerte para detecciones validadas
                small = cv2.resize(face_region, (8, 8), interpolation=cv2.INTER_LINEAR)
            else:
                # Pixelación estándar para detecciones básicas
                small = cv2.resize(face_region, (12, 12), interpolation=cv2.INTER_LINEAR)
            
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Reemplazar región
            frame[y:y+h, x:x+w] = pixelated
            
            # Dibujar rectángulo con colores diferentes
            if method == "TRAINED":
                color = (0, 255, 0)  # Verde para validadas
            else:
                color = (0, 255, 255)  # Amarillo para básicas
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{method} {conf:.2f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Actualizar métricas
        self.face_count = len(validated_faces)
        self.confidence_scores = [conf for _, _, _, _, conf, _ in validated_faces]
        self._update_frame_time(start_time)
        
        return frame

    def _update_frame_time(self, start_time):
        """Actualiza el tiempo de frame para calcular FPS"""
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        # Mantener solo los últimos 30 frames para calcular FPS promedio
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
    
    def get_current_fps(self):
        """Calcula FPS actual basado en los tiempos de frame recientes"""
        if len(self.frame_times) < 2:
            return 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return round(1.0 / avg_frame_time, 1) if avg_frame_time > 0 else 0
    
    def get_average_confidence(self):
        """Calcula confianza promedio de las detecciones recientes"""
        if not self.confidence_scores:
            return 0
        return round(sum(self.confidence_scores) / len(self.confidence_scores) * 100, 1)
    
    def non_max_suppression(self, detections, overlap_threshold=0.3):
        """Supresión de no-máximos para eliminar detecciones duplicadas"""
        if len(detections) == 0:
            return []
        
        # Ordenar por confianza
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        selected = []
        while detections:
            current = detections.pop(0)
            selected.append(current)
            
            # Eliminar detecciones que se superponen mucho
            detections = [det for det in detections if self.calculate_overlap(current, det) < overlap_threshold]
        
        return selected
    
    def calculate_overlap(self, det1, det2):
        """Calcula el overlap entre dos detecciones"""
        x1, y1, w1, h1, _ = det1
        x2, y2, w2, h2, _ = det2
        
        # Calcular intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def load_reference_objects(self):
        """Carga imágenes de referencia específicas subidas por el usuario"""
        print("DEBUG: Cargando objetos de referencia específicos...")
        self.reference_objects = {}
        
        # Cargar objetos específicos desde la carpeta uploads/referencias/
        ref_path = "uploads/referencias/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path, exist_ok=True)
            print("DEBUG: Carpeta de referencias creada")
            return False
        
        # Buscar todas las imágenes en la carpeta de referencias
        ref_files = glob.glob(os.path.join(ref_path, "*.jpg")) + \
                   glob.glob(os.path.join(ref_path, "*.jpeg")) + \
                   glob.glob(os.path.join(ref_path, "*.png"))
        
        print(f"DEBUG: Encontrados {len(ref_files)} archivos de referencia")
        
        if ref_files:
            for img_path in ref_files:
                print(f"DEBUG: Procesando archivo: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Aplicar el mismo procesamiento que en detect_objects_sift
                    img_enhanced = cv2.equalizeHist(img)
                    kp, des = self.sift.detectAndCompute(img_enhanced, None)
                    
                    if des is None:
                        # Intentar con imagen original
                        kp, des = self.sift.detectAndCompute(img, None)
                    
                    if des is not None and len(kp) >= 4:  # Mismo umbral que detect_objects_sift
                        # Usar el nombre del archivo como identificador del objeto específico
                        obj_name = os.path.splitext(os.path.basename(img_path))[0]
                        self.reference_objects[obj_name] = (kp, des, img_path)
                        print(f"DEBUG: Cargado objeto específico: {obj_name} con {len(kp)} keypoints")
                    else:
                        print(f"DEBUG: No se pudieron extraer suficientes características SIFT de {img_path}")
                else:
                    print(f"DEBUG: No se pudo cargar imagen {img_path}")
        
        print(f"DEBUG: Total de objetos específicos cargados: {len(self.reference_objects)}")
        for obj_name in self.reference_objects.keys():
            print(f"DEBUG: - {obj_name}")
        
        return len(self.reference_objects) > 0
    
    def add_reference_object(self, image_path, object_name):
        """Agrega un objeto de referencia específico"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "No se pudo cargar la imagen"
        
        print(f"DEBUG: Procesando imagen de referencia '{object_name}' de tamaño {img.shape}")
        
        # Aplicar exactamente la misma mejora que usamos en detect_objects_sift
        img_enhanced = cv2.equalizeHist(img)
        
        # Extraer características SIFT de la imagen mejorada
        kp, des = self.sift.detectAndCompute(img_enhanced, None)
        if des is None:
            # Intentar con la imagen original si la mejorada falla
            print("DEBUG: Intentando con imagen original...")
            kp, des = self.sift.detectAndCompute(img, None)
            if des is None:
                return False, "No se pudieron extraer características SIFT de la imagen. Intenta con una imagen más detallada o con mejor contraste."
        
        print(f"DEBUG: Extraídos {len(kp)} keypoints SIFT de '{object_name}'")
        
        # Umbral más permisivo basado en el test exitoso
        if len(kp) < 4:  # Mismo umbral mínimo que usamos en detect_objects_sift
            return False, f"La imagen tiene muy pocas características distintivas ({len(kp)} keypoints). Necesita al menos 4 para una detección básica."
        
        # Guardar en la carpeta de referencias
        ref_path = "uploads/referencias/"
        os.makedirs(ref_path, exist_ok=True)
        
        ref_filename = f"{object_name}.jpg"
        ref_filepath = os.path.join(ref_path, ref_filename)
        
        # Guardar imagen de referencia mejorada (la misma que usaremos para matching)
        cv2.imwrite(ref_filepath, img_enhanced)
        
        # Agregar a objetos de referencia
        self.reference_objects[object_name] = (kp, des, ref_filepath)
        
        print(f"DEBUG: Objeto '{object_name}' agregado exitosamente con {len(kp)} keypoints")
        
        return True, f"Objeto '{object_name}' agregado exitosamente con {len(kp)} características SIFT"
    
    def detect_objects_sift(self, frame):
        """Detecta objetos específicos usando SIFT y también detecta rostros automáticamente"""
        # PRIMERO: Detectar y anonimizar rostros automáticamente cuando SIFT está activo
        if self.is_trained:
            frame = self.detect_faces(frame)
        
        # SEGUNDO: Detectar objetos específicos si está configurado
        if not self.detect_objects or not self.reference_objects:
            return frame
        
        # Convertir frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar mejora de contraste similar a las referencias
        gray_enhanced = cv2.equalizeHist(gray)
        
        # Extraer características SIFT del frame mejorado
        kp_frame, des_frame = self.sift.detectAndCompute(gray_enhanced, None)
        
        if des_frame is None:
            print("DEBUG: No se pudieron extraer características SIFT del frame")
            return frame
        
        print(f"DEBUG: Frame tiene {len(kp_frame)} keypoints SIFT")
        
        detections = []
        
        # Buscar cada objeto específico
        for obj_name, (ref_kp, ref_des, ref_path) in self.reference_objects.items():
            print(f"DEBUG: Buscando objeto '{obj_name}' con {len(ref_kp)} keypoints de referencia")
            
            try:
                # Hacer matching exactamente como en el test exitoso
                matches = self.matcher.knnMatch(ref_des, des_frame, k=2)
                
                # Filtrar buenos matches usando Lowe's ratio test (igual que el test)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.8 * n.distance:  # Mismo ratio que el test
                            good_matches.append(m)
                
                print(f"DEBUG: Objeto '{obj_name}' tiene {len(good_matches)} buenos matches")
                
                # Usar el mismo umbral que el test exitoso
                if len(good_matches) >= 4:  # Mismo umbral que el test
                    # Obtener puntos correspondientes
                    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Encontrar homografía con los mismos parámetros que el test
                    try:
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            # Obtener dimensiones de la imagen de referencia
                            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                            if ref_img is not None:
                                h, w = ref_img.shape
                                corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
                                transformed_corners = cv2.perspectiveTransform(corners, M)
                                
                                # Calcular bounding box exactamente como el test
                                x_coords = transformed_corners[:, 0, 0]
                                y_coords = transformed_corners[:, 0, 1]
                                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                                
                                print(f"DEBUG: Bounding box calculado: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                                
                                # Validar que el bounding box sea razonable (menos restrictivo)
                                if (x_max > x_min and y_max > y_min and 
                                    x_min >= -10 and y_min >= -10 and  # Permitir detecciones parciales
                                    x_max <= frame.shape[1] + 10 and y_max <= frame.shape[0] + 10 and
                                    (x_max - x_min) > 10 and (y_max - y_min) > 10):  # Tamaño mínimo más pequeño
                                    
                                    # Asegurar que las coordenadas estén dentro del frame
                                    x_min = max(0, x_min)
                                    y_min = max(0, y_min)
                                    x_max = min(frame.shape[1], x_max)
                                    y_max = min(frame.shape[0], y_max)
                                    
                                    # Calcular confianza basada en inliers como el test
                                    if mask is not None:
                                        inliers = np.sum(mask)
                                        confidence = inliers / len(good_matches)  # Proporción de inliers
                                    else:
                                        confidence = 0.5
                                    
                                    print(f"DEBUG: Objeto '{obj_name}' detectado con confianza {confidence:.2f} (inliers: {np.sum(mask) if mask is not None else 'N/A'})")
                                    
                                    # Umbral de confianza más permisivo
                                    if confidence > 0.3:
                                        detections.append((x_min, y_min, x_max - x_min, y_max - y_min, 
                                                        confidence, obj_name))
                                else:
                                    print(f"DEBUG: Bounding box inválido para '{obj_name}': ({x_min}, {y_min}) - ({x_max}, {y_max})")
                    except Exception as e:
                        print(f"DEBUG: Error en homografía para '{obj_name}': {e}")
                        continue
                else:
                    print(f"DEBUG: Insuficientes matches para '{obj_name}': {len(good_matches)} < 4")
            except Exception as e:
                print(f"DEBUG: Error procesando objeto '{obj_name}': {e}")
                continue
        
        print(f"DEBUG: Total detecciones válidas: {len(detections)}")
        
        # Procesar objetos específicos detectados (pixelar o solo marcar según configuración)
        for (x, y, w, h, conf, obj_name) in detections:
            print(f"DEBUG: Procesando objeto '{obj_name}' en ({x}, {y}, {w}, {h})")
            
            # Validar coordenadas
            if y + h <= frame.shape[0] and x + w <= frame.shape[1] and x >= 0 and y >= 0:
                if self.anonymize_objects:
                    # Pixelar el objeto (comportamiento anterior)
                    obj_region = frame[y:y+h, x:x+w]
                    
                    if obj_region.shape[0] > 0 and obj_region.shape[1] > 0:
                        # Pixelar
                        small = cv2.resize(obj_region, (8, 8), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Reemplazar región
                        frame[y:y+h, x:x+w] = pixelated
                        
                        print(f"DEBUG: Objeto '{obj_name}' pixelado exitosamente")
                
                # Dibujar rectángulo y etiqueta con el nombre específico del objeto
                color = (255, 0, 255) if self.anonymize_objects else (0, 255, 0)  # Magenta si pixelado, verde si solo marco
                thickness = 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                
                # Texto con estado de anonimización
                label = f'{obj_name} {conf:.2f}'
                if not self.anonymize_objects:
                    label += ' (Marco)'
                
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Actualizar métricas de objetos
        self.object_count = len(detections)
        
        return frame
    
    def process_static_image(self, image_path):
        """Procesa una imagen estática para detectar rostros y objetos"""
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: No se pudo cargar la imagen"
        
        print(f"DEBUG: Procesando imagen estática de tamaño: {image.shape}")
        
        # Si tenemos objetos de referencia, detect_objects_sift ya incluye detección de rostros
        if self.detect_objects and self.reference_objects:
            print("DEBUG: Aplicando detección combinada (rostros + objetos SIFT) a imagen estática")
            image = self.detect_objects_sift(image)
        else:
            # Solo detección de rostros si no hay objetos de referencia
            print("DEBUG: Aplicando solo detección de rostros a imagen estática")
            image = self.detect_faces(image)
        
        return image, "Procesamiento exitoso"

# Instancia global del detector
detector = FaceDetectorLBP()

# Intentar cargar modelo previamente entrenado
if not detector.load_trained_model():
    print("No se encontró modelo entrenado. Necesitas entrenar primero.")

# Variable para controlar la cámara
camera = None
camera_active = False

def get_camera():
    """Obtiene la cámara"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    """Genera frames para el streaming de video"""
    global camera_active
    camera = get_camera()
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Si tenemos objetos de referencia, detect_objects_sift ya incluye detección de rostros
        if detector.detect_objects and detector.reference_objects:
            # Detección combinada (rostros + objetos SIFT)
            frame = detector.detect_objects_sift(frame)
        else:
            # Solo detección de rostros si no hay objetos de referencia
            frame = detector.detect_faces(frame)
        
        # Codificar frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/train')
def train():
    """Inicia el entrenamiento del clasificador"""
    def train_thread():
        detector.train_classifier()
    
    if not detector.is_trained:
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        return jsonify({"status": "training_started"})
    else:
        return jsonify({"status": "already_trained"})

@app.route('/training_progress')
def training_progress():
    """Obtiene el progreso del entrenamiento"""
    return jsonify({
        "progress": detector.training_progress,
        "is_trained": detector.is_trained
    })

@app.route('/video_feed')
def video_feed():
    """Stream de video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Inicia la cámara"""
    global camera_active
    camera_active = True
    return jsonify({"status": "camera_started"})

@app.route('/stop_camera')
def stop_camera():
    """Detiene la cámara"""
    global camera_active, camera
    camera_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "camera_stopped"})

@app.route('/load_objects')
def load_objects():
    """Carga las imágenes de referencia específicas para detección SIFT"""
    success = detector.load_reference_objects()
    if success:
        return jsonify({
            "status": "objects_loaded",
            "objects": list(detector.reference_objects.keys()),
            "count": len(detector.reference_objects)
        })
    else:
        return jsonify({"status": "no_objects_found"})

@app.route('/add_reference_object', methods=['POST'])
def add_reference_object():
    """Sube una imagen de referencia específica para un objeto"""
    if 'file' not in request.files or 'object_name' not in request.form:
        return jsonify({"status": "error", "message": "Faltan archivo o nombre del objeto"})
    
    file = request.files['file']
    object_name = request.form['object_name'].strip()
    
    if file.filename == '' or object_name == '':
        return jsonify({"status": "error", "message": "Selecciona un archivo y proporciona un nombre"})
    
    if not object_name.replace('_', '').replace('-', '').isalnum():
        return jsonify({"status": "error", "message": "El nombre solo puede contener letras, números, guiones y guiones bajos"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        temp_filename = f"temp_{timestamp}_{filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)
        
        # Agregar como objeto de referencia
        success, message = detector.add_reference_object(temp_filepath, object_name)
        
        # Limpiar archivo temporal
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        if success:
            return jsonify({
                "status": "success",
                "message": message,
                "object_name": object_name
            })
        else:
            return jsonify({"status": "error", "message": message})
    
    return jsonify({"status": "error", "message": "Formato de archivo no válido"})

@app.route('/get_reference_objects')
def get_reference_objects():
    """Obtiene la lista de objetos de referencia cargados"""
    objects = []
    for obj_name, (kp, des, ref_path) in detector.reference_objects.items():
        # Leer imagen para convertir a base64
        img = cv2.imread(ref_path)
        if img is not None:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            objects.append({
                "name": obj_name,
                "image_data": img_base64,
                "keypoints": len(kp)
            })
    
    return jsonify({
        "status": "success",
        "objects": objects,
        "count": len(objects)
    })

@app.route('/delete_reference_object/<object_name>')
def delete_reference_object(object_name):
    """Elimina un objeto de referencia específico"""
    if object_name in detector.reference_objects:
        # Eliminar archivo de referencia
        ref_path = detector.reference_objects[object_name][2]
        if os.path.exists(ref_path):
            os.remove(ref_path)
        
        # Eliminar de memoria
        del detector.reference_objects[object_name]
        
        return jsonify({
            "status": "success",
            "message": f"Objeto '{object_name}' eliminado exitosamente"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Objeto no encontrado"
        })

@app.route('/toggle_objects')
def toggle_objects():
    """Activa/desactiva la detección de objetos"""
    detector.detect_objects = not detector.detect_objects
    return jsonify({
        "status": "objects_toggled",
        "detect_objects": detector.detect_objects
    })

@app.route('/set_anonymization_mode', methods=['POST'])
def set_anonymization_mode():
    """Configura el modo de anonimización de objetos"""
    try:
        data = request.get_json()
        anonymize = data.get('anonymize', True)
        detector.anonymize_objects = anonymize
        
        print(f"DEBUG: Modo de anonimización establecido a: {anonymize}")
        
        return jsonify({
            "status": "success",
            "anonymize_objects": detector.anonymize_objects
        })
    except Exception as e:
        print(f"ERROR: No se pudo establecer modo de anonimización: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/object_status')
def object_status():
    """Obtiene el estado de la detección de objetos"""
    return jsonify({
        "detect_objects": detector.detect_objects,
        "anonymize_objects": detector.anonymize_objects,
        "objects_loaded": len(detector.reference_objects) > 0,
        "available_objects": list(detector.reference_objects.keys()),
        "count": len(detector.reference_objects)
    })

@app.route('/get_system_metrics')
def get_system_metrics():
    """Obtiene métricas del sistema en tiempo real"""
    try:
        # Obtener métricas de memoria y CPU
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Obtener métricas del detector
        fps = detector.get_current_fps()
        confidence = detector.get_average_confidence()
        
        return jsonify({
            "status": "success",
            "fps": f"{fps} FPS",
            "memory": f"{memory_info.used // (1024*1024)} MB",
            "cpu": f"{cpu_percent:.1f}%",
            "confidence": f"{confidence:.1f}%",
            "faces_detected": detector.face_count,
            "objects_detected": detector.object_count
        })
    except Exception as e:
        return jsonify({
            "status": "success",  # Fallback graceful
            "fps": "-- FPS",
            "memory": "-- MB", 
            "cpu": "--%",
            "confidence": "--%",
            "faces_detected": detector.face_count,
            "objects_detected": detector.object_count
        })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Sube y procesa una imagen para detectar objetos"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No se seleccionó archivo"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No se seleccionó archivo"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Agregar timestamp para evitar conflictos
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Procesar imagen
        processed_image, message = detector.process_static_image(filepath)
        
        if processed_image is not None:
            # Guardar imagen procesada
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_image)
            
            # Convertir imagen a base64 para mostrar en web
            _, buffer = cv2.imencode('.jpg', processed_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "status": "success", 
                "message": message,
                "image_data": img_base64,
                "processed_filename": processed_filename
            })
        else:
            return jsonify({"status": "error", "message": message})
    
    return jsonify({"status": "error", "message": "Formato de archivo no válido"})

@app.route('/download_processed/<filename>')
def download_processed(filename):
    """Descarga la imagen procesada"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except:
        return jsonify({"status": "error", "message": "Archivo no encontrado"})

if __name__ == '__main__':
    # Crear directorio de templates si no existe
    os.makedirs("templates", exist_ok=True)
    
    print("=== DETECTOR DE ROSTROS Y OBJETOS ESPECÍFICOS ===")
    print("SECCIÓN 1: Detección de rostros con SVM+HOG")
    print("SECCIÓN 2: Identificación de objetos específicos con SIFT")
    print("")
    print("FUNCIONALIDADES PRINCIPALES:")
    print("• Detección y pixelación de rostros en tiempo real")
    print("• Subida de objetos de referencia específicos")
    print("• Identificación exacta de esos objetos en otras imágenes")
    print("• Ejemplo: Sube foto del perro 'Rex', luego busca a 'Rex' en fotos familiares")
    print("")
    print("1. Navega a http://localhost:5000")
    print("2. Entrena el clasificador de rostros")
    print("3. Agrega objetos específicos como referencia")
    print("4. Sube imágenes para buscar esos objetos específicos")
    print("===============================================")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
