# === IMPORTACIONES DE LIBRERÍAS ===
import cv2  # OpenCV para procesamiento de imágenes y detección de características
import os  # Manejo de archivos y directorios del sistema operativo
import numpy as np  # Cálculos numéricos y manejo de arrays multidimensionales
from flask import Flask, render_template, Response, jsonify, request, send_file  # Framework web para crear la aplicación
import threading  # Manejo de hilos para ejecutar tareas en paralelo
import time  # Funciones relacionadas con tiempo y delays
from sklearn.svm import SVC  # Máquina de Vectores de Soporte para clasificación
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Métrica para evaluar precisión del modelo
import pickle  # Serialización de objetos Python para guardar/cargar modelos
from skimage.feature import hog  # Extracción de características HOG (Histogram of Oriented Gradients)
import random  # Generación de números aleatorios
import glob  # Búsqueda de archivos con patrones específicos
from werkzeug.utils import secure_filename  # Seguridad para nombres de archivos subidos
import base64  # Codificación/decodificación base64 para imágenes
from io import BytesIO  # Manejo de datos binarios en memoria
from PIL import Image  # Procesamiento de imágenes con Pillow
import psutil  # Monitoreo de recursos del sistema (CPU, memoria)

# === CONFIGURACIÓN DE LA APLICACIÓN FLASK ===
app = Flask(__name__)  # Crear instancia de la aplicación Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite máximo de archivo: 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'  # Carpeta donde se guardarán los archivos subidos

# === INICIALIZACIÓN DE DIRECTORIOS ===
os.makedirs('uploads', exist_ok=True)  # Crear carpeta 'uploads' si no existe

# === VARIABLES GLOBALES PARA MONITOREO DEL SISTEMA ===
system_metrics = {
    'fps': 0,  # Frames por segundo del video en tiempo real
    'memory': '0 MB',  # Uso de memoria RAM del sistema
    'cpu': '0%',  # Porcentaje de uso de CPU
    'confidence': '0%',  # Confianza promedio de las detecciones
    'faces_detected': 0,  # Número de rostros detectados en el frame actual
    'objects_detected': 0,  # Número de objetos específicos detectados
    'last_update': time.time()  # Timestamp de la última actualización de métricas
}

# === CONFIGURACIÓN DE ARCHIVOS PERMITIDOS ===
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}  # Formatos de imagen válidos

def allowed_file(filename):
    """
    Verifica si un archivo tiene una extensión válida
    Args:
        filename (str): Nombre del archivo a verificar
    Returns:
        bool: True si la extensión es válida, False en caso contrario
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  # Obtener extensión y verificar si está permitida

# === CLASE PRINCIPAL PARA DETECCIÓN DE ROSTROS Y OBJETOS ===
class FaceDetectorLBP:
    """
    Clase principal que maneja:
    1. Detección de rostros usando HOG + SVM
    2. Detección de objetos específicos usando SIFT
    3. Anonimización mediante pixelación
    """
    def __init__(self):
        # === VARIABLES PARA CLASIFICACIÓN DE ROSTROS ===
        self.classifier = None  # Clasificador de OpenCV (no usado actualmente)
        self.svm_model = None  # Modelo SVM entrenado para detectar rostros
        self.is_trained = False  # Flag que indica si el modelo está entrenado
        self.training_progress = 0  # Progreso del entrenamiento (0-100%)
        self.window_size = (64, 64)  # Tamaño estándar para extraer características HOG
        
        # === VARIABLES PARA DETECCIÓN DE OBJETOS ESPECÍFICOS ===
        self.sift = cv2.SIFT_create()  # Detector SIFT para encontrar puntos clave
        self.matcher = cv2.BFMatcher()  # Matcher para comparar características SIFT
        self.reference_objects = {}  # Diccionario que almacena objetos de referencia subidos por el usuario
        self.detect_objects = False  # Flag para activar/desactivar detección de objetos específicos
        self.anonymize_objects = True  # Flag para controlar si pixelar objetos o solo marcarlos con rectángulo
        
        # === VARIABLES PARA MÉTRICAS DE RENDIMIENTO ===
        self.frame_times = []  # Lista de tiempos de procesamiento por frame
        self.face_count = 0  # Contador de rostros detectados en el frame actual
        self.object_count = 0  # Contador de objetos específicos detectados
        self.confidence_scores = []  # Lista de scores de confianza de las detecciones
        
    def extract_hog_features(self, image):
        """
        Extrae características HOG (Histogram of Oriented Gradients) de una imagen
        HOG es un descriptor de características que cuenta la orientación de gradientes
        en regiones localizadas de una imagen, útil para detección de objetos
        
        Args:
            image: Imagen de entrada (puede ser color o escala de grises)
        Returns:
            array: Vector de características HOG de dimensión fija
        """
        # Redimensionar imagen al tamaño estándar de ventana (64x64 píxeles)
        image_resized = cv2.resize(image, self.window_size)
        
        # Convertir a escala de grises si la imagen es a color (tiene 3 canales)
        if len(image_resized.shape) == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)  # BGR a escala de grises
        else:
            image_gray = image_resized  # Ya está en escala de grises
            
        # Extraer características HOG con parámetros específicos:
        # - orientations=9: 9 bins para direcciones de gradientes (0-180°)
        # - pixels_per_cell=(8,8): cada celda de 8x8 píxeles
        # - cells_per_block=(2,2): cada bloque contiene 2x2 celdas para normalización
        # - visualize=False: no generar imagen de visualización
        features = hog(image_gray, 
                      orientations=9, 
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), 
                      visualize=False)
        return features  # Retorna vector de características de dimensión fija
    
    def load_dataset(self):
        """
        Carga y procesa el dataset completo para entrenar el clasificador de rostros
        Lee imágenes positivas (rostros) y negativas (no-rostros) desde directorios específicos
        
        Returns:
            tuple: (features, labels) - arrays de numpy con características y etiquetas
        """
        print("Cargando dataset...")
        
        features = []  # Lista para almacenar vectores de características HOG
        labels = []    # Lista para almacenar etiquetas (1=rostro, 0=no-rostro)
        
        # === CARGAR IMÁGENES POSITIVAS (ROSTROS) ===
        positive_path = "dataset/positive/person"  # Ruta donde están las imágenes de rostros
        positive_count = 0  # Contador de imágenes positivas procesadas
        
        if os.path.exists(positive_path):  # Verificar que el directorio existe
            for img_file in os.listdir(positive_path):  # Iterar sobre todos los archivos
                # Verificar que el archivo tenga extensión de imagen válida
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(positive_path, img_file)  # Ruta completa del archivo
                    try:
                        image = cv2.imread(img_path)  # Cargar imagen con OpenCV
                        if image is not None:  # Verificar que la imagen se cargó correctamente
                            # Extraer características HOG de la imagen
                            hog_features = self.extract_hog_features(image)
                            features.append(hog_features)  # Agregar características a la lista
                            labels.append(1)  # Etiqueta 1 = rostro (clase positiva)
                            positive_count += 1  # Incrementar contador
                            
                            # Mostrar progreso cada 100 imágenes procesadas
                            if positive_count % 100 == 0:
                                print(f"Procesadas {positive_count} imágenes positivas...")
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
                        continue  # Continuar con la siguiente imagen si hay error
        
        print(f"Total imágenes positivas procesadas: {positive_count}")
        
        # === CARGAR IMÁGENES NEGATIVAS (NO-ROSTROS) ===
        negative_path = "dataset/negative"  # Ruta donde están las imágenes sin rostros
        negative_count = 0  # Contador de imágenes negativas procesadas
        max_negatives = positive_count * 2  # Usar el doble de imágenes negativas para balancear dataset
        # === RECOPILAR TODAS LAS IMÁGENES NEGATIVAS ===
        negative_files = []  # Lista para almacenar rutas de archivos negativos
        
        # Buscar en todas las subcategorías del directorio negative
        for category in os.listdir(negative_path):  # Iterar sobre subcarpetas (ej: animals, objects, etc.)
            category_path = os.path.join(negative_path, category)  # Ruta completa de la subcarpeta
            if os.path.isdir(category_path):  # Verificar que es un directorio
                # Buscar imágenes en la subcarpeta
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        negative_files.append(os.path.join(category_path, img_file))  # Agregar ruta completa
        
        # === BALANCEAR EL DATASET ===
        random.shuffle(negative_files)  # Mezclar aleatoriamente las imágenes negativas
        negative_files = negative_files[:max_negatives]  # Limitar al número máximo calculado
        
        # === PROCESAR IMÁGENES NEGATIVAS ===
        for img_path in negative_files:  # Iterar sobre las imágenes negativas seleccionadas
            try:
                image = cv2.imread(img_path)  # Cargar imagen
                if image is not None:  # Verificar que se cargó correctamente
                    # Extraer características HOG de la imagen negativa
                    hog_features = self.extract_hog_features(image)
                    features.append(hog_features)  # Agregar características a la lista
                    labels.append(0)  # Etiqueta 0 = no-rostro (clase negativa)
                    negative_count += 1  # Incrementar contador
                    
                    # Mostrar progreso cada 100 imágenes procesadas
                    if negative_count % 100 == 0:
                        print(f"Procesadas {negative_count} imágenes negativas...")
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue  # Continuar con la siguiente imagen si hay error
        
        # === RESUMEN DEL DATASET CARGADO ===
        print(f"Total imágenes negativas procesadas: {negative_count}")
        print(f"Dataset total: {len(features)} imágenes")
        
        # Convertir listas a arrays de numpy para compatibilidad con scikit-learn
        return np.array(features), np.array(labels)
    
    def train_classifier(self):
        """
        Entrena el clasificador SVM usando características HOG extraídas del dataset
        Este es el proceso principal de machine learning del sistema
        
        Returns:
            bool: True si el entrenamiento fue exitoso, False en caso contrario
        """
        print("=== INICIANDO ENTRENAMIENTO REAL ===")
        print("Esto puede tomar varios minutos...")
        
        try:
            # === PASO 1: CARGAR Y PROCESAR DATASET ===
            self.training_progress = 10  # Actualizar progreso para la interfaz web
            features, labels = self.load_dataset()  # Cargar imágenes y extraer características HOG
            
            # Verificar que se cargaron datos
            if len(features) == 0:
                print("Error: No se pudieron cargar imágenes del dataset")
                return False
            
            self.training_progress = 30  # Actualizar progreso
            
            # === PASO 2: DIVIDIR DATASET EN ENTRENAMIENTO Y PRUEBA ===
            print("Dividiendo dataset...")
            # Usar 80% para entrenamiento, 20% para prueba
            # stratify=labels asegura que ambos conjuntos tengan proporción similar de clases
            # random_state=42 hace que la división sea reproducible
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            self.training_progress = 50  # Actualizar progreso
            
            # === PASO 3: ENTRENAR MODELO SVM ===
            print("Entrenando modelo SVM...")
            print(f"Datos de entrenamiento: {len(X_train)} muestras")
            print(f"Datos de prueba: {len(X_test)} muestras")
            
            # Crear y configurar modelo SVM:
            # - kernel='rbf': usa kernel radial (gaussiano) para separación no-lineal
            # - C=1.0: parámetro de regularización (controla trade-off entre margen y errores)
            # - gamma='scale': parámetro del kernel RBF (valor automático basado en características)
            # - probability=True: habilita cálculo de probabilidades para confianza
            self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            self.svm_model.fit(X_train, y_train)  # Entrenar el modelo con datos de entrenamiento
            
            self.training_progress = 80  # Actualizar progreso
            
            # === PASO 4: EVALUAR RENDIMIENTO DEL MODELO ===
            print("Evaluando modelo...")
            y_pred = self.svm_model.predict(X_test)  # Hacer predicciones en datos de prueba
            accuracy = accuracy_score(y_test, y_pred)  # Calcular precisión (accuracy)
            print(f"Precisión del modelo: {accuracy:.2%}")
            
            self.training_progress = 90  # Actualizar progreso
            
            # === PASO 5: GUARDAR MODELO ENTRENADO ===
            print("Guardando modelo...")
            os.makedirs("trained_model", exist_ok=True)  # Crear carpeta si no existe
            # Serializar modelo usando pickle para poder cargarlo después
            with open("trained_model/face_detector_svm.pkl", "wb") as f:
                pickle.dump(self.svm_model, f)
            
            # === FINALIZAR ENTRENAMIENTO ===
            self.is_trained = True  # Marcar que el modelo está listo
            self.training_progress = 100  # Entrenamiento completado
            
            print("=== ENTRENAMIENTO COMPLETADO ===")
            print(f"Modelo entrenado con {len(features)} imágenes")
            print(f"Precisión: {accuracy:.2%}")
            print("Modelo guardado en: trained_model/face_detector_svm.pkl")
            
            return True  # Entrenamiento exitoso
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            return False  # Entrenamiento falló
    
    def load_trained_model(self):
        """
        Carga un modelo SVM previamente entrenado desde disco
        Útil para evitar re-entrenar cada vez que se inicia la aplicación
        
        Returns:
            bool: True si el modelo se cargó exitosamente, False en caso contrario
        """
        model_path = "trained_model/face_detector_svm.pkl"  # Ruta del modelo guardado
        
        if os.path.exists(model_path):  # Verificar que el archivo existe
            try:
                # Deserializar modelo usando pickle
                with open(model_path, "rb") as f:
                    self.svm_model = pickle.load(f)
                
                # Actualizar estados internos
                self.is_trained = True  # Marcar que el modelo está listo
                self.training_progress = 100  # Progreso completo
                print("Modelo entrenado cargado exitosamente")
                return True  # Carga exitosa
                
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
        return False  # No se pudo cargar
    
    def detect_faces(self, frame):
        """
        Detecta rostros usando un enfoque híbrido de dos etapas:
        1. Detección rápida con cascadas Haar de OpenCV
        2. Validación con modelo SVM entrenado (si está disponible)
        
        Args:
            frame: Frame de video o imagen donde detectar rostros
        Returns:
            frame: Frame modificado con rostros pixelados y marcados
        """
        start_time = time.time()  # Iniciar cronómetro para medir rendimiento
        
        # === PASO 1: DETECCIÓN RÁPIDA CON OPENCV ===
        # Cargar clasificador de cascadas Haar pre-entrenado para rostros frontales
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        
        # Detectar rostros candidatos:
        # - scaleFactor=1.1: factor de escala entre niveles de imagen (más pequeño = más preciso pero más lento)
        # - minNeighbors=4: número mínimo de detecciones vecinas requeridas
        # - minSize=(50,50): tamaño mínimo de rostro en píxeles
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        validated_faces = []  # Lista para almacenar rostros validados
        
        # === PASO 2: VALIDACIÓN CON MODELO SVM ENTRENADO ===
        if self.is_trained and self.svm_model is not None:  # Si tenemos modelo entrenado disponible
            for (x, y, w, h) in faces:  # Iterar sobre cada rostro candidato detectado
                # Extraer región rectangular del rostro candidato
                face_region = frame[y:y+h, x:x+w]
                
                try:
                    # Extraer características HOG de la región del rostro
                    hog_features = self.extract_hog_features(face_region)
                    
                    # Hacer predicción con nuestro modelo SVM entrenado
                    prediction = self.svm_model.predict([hog_features])[0]  # Clase predicha (0 o 1)
                    probability = self.svm_model.predict_proba([hog_features])[0]  # Probabilidades de cada clase
                    
                    # Decidir si es un rostro válido basado en predicción y confianza
                    if prediction == 1 and probability[1] > 0.6:  # Umbral de confianza del 60%
                        validated_faces.append((x, y, w, h, probability[1], "TRAINED"))  # Rostro validado por SVM
                    else:
                        # Si SVM no está seguro, usar detección básica con menor confianza
                        validated_faces.append((x, y, w, h, 0.8, "BASIC"))
                        
                except:
                    # Si hay error en el procesamiento SVM, usar detección básica
                    validated_faces.append((x, y, w, h, 0.8, "BASIC"))
        else:
            # === FALLBACK: SOLO DETECCIÓN BÁSICA ===
            # Si no tenemos modelo entrenado, usar todas las detecciones de OpenCV
            for (x, y, w, h) in faces:
                validated_faces.append((x, y, w, h, 0.8, "BASIC"))
        # === PASO 3: ANONIMIZACIÓN MEDIANTE PIXELACIÓN ===
        for (x, y, w, h, conf, method) in validated_faces:  # Procesar cada rostro validado
            # Extraer región rectangular del rostro del frame original
            face_region = frame[y:y+h, x:x+w]
            
            # === APLICAR PIXELACIÓN CON DIFERENTES INTENSIDADES ===
            if method == "TRAINED":
                # Pixelación más agresiva para rostros validados por SVM (más confiables)
                # Reducir a 8x8 píxeles para mayor anonimización
                small = cv2.resize(face_region, (8, 8), interpolation=cv2.INTER_LINEAR)
            else:
                # Pixelación moderada para detecciones básicas (menos confiables)
                # Reducir a 12x12 píxeles para preservar algo más de detalle
                small = cv2.resize(face_region, (12, 12), interpolation=cv2.INTER_LINEAR)
            
            # Redimensionar de vuelta al tamaño original con interpolación pixelada
            # INTER_NEAREST mantiene el efecto de bloques/píxeles grandes
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # === REEMPLAZAR REGIÓN ORIGINAL CON VERSIÓN PIXELADA ===
            frame[y:y+h, x:x+w] = pixelated
            
            # === DIBUJAR MARCADORES VISUALES ===
            # Usar colores diferentes para distinguir tipos de detección
            if method == "TRAINED":
                color = (0, 255, 0)  # Verde para rostros validados por SVM
            else:
                color = (0, 255, 255)  # Amarillo para detecciones básicas
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Agregar etiqueta con tipo de detección y confianza
            cv2.putText(frame, f'{method} {conf:.2f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # === ACTUALIZAR MÉTRICAS DE RENDIMIENTO ===
        self.face_count = len(validated_faces)  # Número de rostros detectados
        self.confidence_scores = [conf for _, _, _, _, conf, _ in validated_faces]  # Scores de confianza
        self._update_frame_time(start_time)  # Actualizar tiempo de procesamiento para FPS
        
        return frame  # Retornar frame procesado con rostros anonimizados

    def _update_frame_time(self, start_time):
        """
        Actualiza la lista de tiempos de procesamiento para calcular FPS promedio
        Mantiene un historial deslizante de los últimos 30 frames
        
        Args:
            start_time: Timestamp de cuando comenzó el procesamiento del frame
        """
        frame_time = time.time() - start_time  # Calcular tiempo transcurrido
        self.frame_times.append(frame_time)  # Agregar a la lista de tiempos
        
        # Mantener solo los últimos 30 frames para calcular FPS promedio
        # Esto evita que la lista crezca indefinidamente y proporciona métricas más actuales
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)  # Eliminar el tiempo más antiguo
    
    def get_current_fps(self):
        """
        Calcula FPS (Frames Per Second) actual basado en los tiempos de frame recientes
        
        Returns:
            float: FPS promedio redondeado a 1 decimal, 0 si no hay suficientes datos
        """
        if len(self.frame_times) < 2:  # Necesitamos al menos 2 frames para calcular FPS
            return 0
        
        # Calcular tiempo promedio por frame
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # FPS = 1 / tiempo_por_frame, con protección contra división por cero
        return round(1.0 / avg_frame_time, 1) if avg_frame_time > 0 else 0
    
    def get_average_confidence(self):
        """
        Calcula la confianza promedio de las detecciones recientes
        
        Returns:
            float: Confianza promedio como porcentaje (0-100%), 0 si no hay detecciones
        """
        if not self.confidence_scores:  # Si no hay scores de confianza
            return 0
        
        # Calcular promedio de confianzas y convertir a porcentaje
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        return round(avg_confidence * 100, 1)  # Multiplicar por 100 para porcentaje
    
    def non_max_suppression(self, detections, overlap_threshold=0.3):
        """
        Implementa supresión de no-máximos para eliminar detecciones duplicadas/superpuestas
        Algoritmo usado en detección de objetos para limpiar resultados
        
        Args:
            detections: Lista de detecciones [(x, y, w, h, confidence, ...)]
            overlap_threshold: Umbral de superposición máxima permitida (0.0-1.0)
        Returns:
            list: Lista filtrada de detecciones sin duplicados
        """
        if len(detections) == 0:  # Si no hay detecciones, retornar lista vacía
            return []
        
        # === ORDENAR POR CONFIANZA (MAYOR A MENOR) ===
        # Las detecciones con mayor confianza tienen prioridad
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        selected = []  # Lista de detecciones seleccionadas (sin duplicados)
        
        # === ALGORITMO DE SUPRESIÓN ===
        while detections:  # Mientras queden detecciones por procesar
            current = detections.pop(0)  # Tomar la detección con mayor confianza restante
            selected.append(current)  # Agregar a seleccionadas
            
            # Eliminar detecciones que se superponen demasiado con la actual
            # Mantener solo las que tienen overlap menor al umbral
            detections = [det for det in detections 
                         if self.calculate_overlap(current, det) < overlap_threshold]
        
        return selected  # Retornar detecciones filtradas
    
    def calculate_overlap(self, det1, det2):
        """
        Calcula el IoU (Intersection over Union) entre dos detecciones rectangulares
        Métrica estándar para medir superposición entre bounding boxes
        
        Args:
            det1: Primera detección (x, y, w, h, confidence, ...)
            det2: Segunda detección (x, y, w, h, confidence, ...)
        Returns:
            float: Valor IoU entre 0.0 (sin superposición) y 1.0 (superposición completa)
        """
        # Extraer coordenadas y dimensiones de ambas detecciones
        x1, y1, w1, h1, _ = det1  # Primera detección
        x2, y2, w2, h2, _ = det2  # Segunda detección
        
        # === CALCULAR COORDENADAS DE LA INTERSECCIÓN ===
        # La intersección es el rectángulo formado por el área común
        xi1 = max(x1, x2)        # Coordenada x izquierda de la intersección
        yi1 = max(y1, y2)        # Coordenada y superior de la intersección
        xi2 = min(x1 + w1, x2 + w2)  # Coordenada x derecha de la intersección
        yi2 = min(y1 + h1, y2 + h2)  # Coordenada y inferior de la intersección
        
        # Verificar si hay intersección real
        if xi2 <= xi1 or yi2 <= yi1:  # No hay superposición
            return 0
        
        # === CALCULAR ÁREAS ===
        intersection = (xi2 - xi1) * (yi2 - yi1)  # Área de intersección
        area1 = w1 * h1  # Área de la primera detección
        area2 = w2 * h2  # Área de la segunda detección
        union = area1 + area2 - intersection  # Área de unión (sin doble conteo)
        
        # === CALCULAR IoU ===
        # IoU = Intersección / Unión
        return intersection / union if union > 0 else 0
    
    def load_reference_objects(self):
        """
        Carga imágenes de referencia específicas subidas por el usuario para detección SIFT
        Estas imágenes servirán como plantillas para buscar objetos específicos en tiempo real
        
        Returns:
            bool: True si se cargaron objetos, False si no se encontraron
        """
        print("DEBUG: Cargando objetos de referencia específicos...")
        self.reference_objects = {}  # Reiniciar diccionario de objetos
        
        # === CONFIGURAR CARPETA DE REFERENCIAS ===
        ref_path = "uploads/referencias/"  # Carpeta donde se guardan las imágenes de referencia
        if not os.path.exists(ref_path):  # Si no existe la carpeta
            os.makedirs(ref_path, exist_ok=True)  # Crearla
            print("DEBUG: Carpeta de referencias creada")
            return False  # No hay objetos que cargar aún
        
        # === BUSCAR ARCHIVOS DE IMAGEN EN LA CARPETA ===
        # Buscar todos los formatos de imagen soportados
        ref_files = glob.glob(os.path.join(ref_path, "*.jpg")) + \
                   glob.glob(os.path.join(ref_path, "*.jpeg")) + \
                   glob.glob(os.path.join(ref_path, "*.png"))
        
        print(f"DEBUG: Encontrados {len(ref_files)} archivos de referencia")
        
        # === PROCESAR CADA IMAGEN DE REFERENCIA ===
        if ref_files:
            for img_path in ref_files:  # Iterar sobre cada archivo encontrado
                print(f"DEBUG: Procesando archivo: {img_path}")
                
                # Cargar imagen en escala de grises (SIFT funciona mejor en grayscale)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:  # Si la imagen se cargó correctamente
                    # === MEJORAR CONTRASTE PARA MEJOR EXTRACCIÓN SIFT ===
                    img_enhanced = cv2.equalizeHist(img)  # Ecualización de histograma
                    
                    # Extraer puntos clave (keypoints) y descriptores SIFT
                    kp, des = self.sift.detectAndCompute(img_enhanced, None)
                    
                    # Si falló con imagen mejorada, intentar con original
                    if des is None:
                        kp, des = self.sift.detectAndCompute(img, None)
                    
                    # === VALIDAR CALIDAD DE LA EXTRACCIÓN ===
                    # Necesitamos al menos 4 keypoints para hacer matching robusto
                    if des is not None and len(kp) >= 4:
                        # Usar nombre del archivo (sin extensión) como identificador
                        obj_name = os.path.splitext(os.path.basename(img_path))[0]
                        
                        # Guardar: (keypoints, descriptores, ruta_archivo)
                        self.reference_objects[obj_name] = (kp, des, img_path)
                        print(f"DEBUG: Cargado objeto específico: {obj_name} con {len(kp)} keypoints")
                    else:
                        print(f"DEBUG: No se pudieron extraer suficientes características SIFT de {img_path}")
                else:
                    print(f"DEBUG: No se pudo cargar imagen {img_path}")
        
        # === RESUMEN DE CARGA ===
        print(f"DEBUG: Total de objetos específicos cargados: {len(self.reference_objects)}")
        for obj_name in self.reference_objects.keys():
            print(f"DEBUG: - {obj_name}")
        
        return len(self.reference_objects) > 0  # True si se cargó al menos un objeto
    
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
        """
        Detecta objetos específicos usando SIFT (Scale-Invariant Feature Transform)
        También ejecuta detección de rostros automáticamente cuando está activo
        
        SIFT es un algoritmo que:
        1. Detecta puntos clave invariantes a escala, rotación e iluminación
        2. Genera descriptores únicos para cada punto clave
        3. Permite encontrar objetos específicos mediante matching de descriptores
        
        Args:
            frame: Frame de video donde buscar objetos y rostros
        Returns:
            frame: Frame procesado con detecciones marcadas/anonimizadas
        """
        
        # === PASO 1: DETECCIÓN AUTOMÁTICA DE ROSTROS ===
        # Siempre detectar rostros cuando SIFT está activo (comportamiento híbrido)
        if self.is_trained:  # Si tenemos modelo de rostros entrenado
            frame = self.detect_faces(frame)  # Aplicar detección y anonimización de rostros
        
        # === PASO 2: VERIFICAR SI PROCEDER CON DETECCIÓN DE OBJETOS ===
        # Solo continuar si la detección de objetos está habilitada y hay referencias cargadas
        if not self.detect_objects or not self.reference_objects:
            return frame  # Retornar frame solo con detección de rostros
        
        # === PASO 3: PREPARAR FRAME PARA SIFT ===
        # Convertir frame a escala de grises (SIFT funciona mejor en grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar mejora de contraste igual que en las referencias
        # Esto asegura consistencia en la extracción de características
        gray_enhanced = cv2.equalizeHist(gray)
        
        # === PASO 4: EXTRAER CARACTERÍSTICAS SIFT DEL FRAME ===
        kp_frame, des_frame = self.sift.detectAndCompute(gray_enhanced, None)
        
        # Verificar que se extrajeron características del frame
        if des_frame is None:
            print("DEBUG: No se pudieron extraer características SIFT del frame")
            return frame  # No hay características para hacer matching
        
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
