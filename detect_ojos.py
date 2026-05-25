import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime as ort

def load_cascade_classifiers():
    """
    Carga los clasificadores Haar Cascade integrados de OpenCV.
    """
    face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
    cat_face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface.xml")
    
    if not os.path.exists(face_path) or not os.path.exists(eye_path) or not os.path.exists(eye_glass_path) or not os.path.exists(cat_face_path):
        raise FileNotFoundError("No se pudieron localizar los archivos Haar Cascade en la instalación de OpenCV.")
        
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)
    cat_face_cascade = cv2.CascadeClassifier(cat_face_path)
    return face_cascade, eye_cascade, eye_glass_cascade, cat_face_cascade

def evaluate_crop_internal(img, x, y, w, h, session, input_name, output_name):
    """
    Extrae, redimensiona y evalúa un parche de imagen utilizando el modelo ONNX.
    """
    img_h, img_w, _ = img.shape
    x1 = max(0, min(x, img_w - 10))
    y1 = max(0, min(y, img_h - 10))
    w1 = max(10, min(w, img_w - x1))
    h1 = max(10, min(h, img_h - y1))
    
    crop = img[y1:y1+h1, x1:x1+w1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224, 224))
    
    input_tensor = crop_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    outputs = session.run([output_name], {input_name: input_tensor})
    score = float(outputs[0][0][0])
    return score, (x1, y1, w1, h1)

def refine_box(img, x, y, w, h, session, input_name, output_name):
    """
    Realiza una búsqueda local alrededor del candidato original para maximizar la puntuación ONNX,
    logrando centrar y ajustar la escala del cuadro exactamente sobre el ojo.
    """
    # Inicializar con el cuadro original
    best_score, best_box = evaluate_crop_internal(img, x, y, w, h, session, input_name, output_name)
    
    # Búsqueda local de cuadrícula
    # ds: variación del tamaño (manteniendo proporción cuadrada)
    for ds in range(-6, 7, 2):
        size = w + ds
        if size < 15:
            continue
        old_cx = x + w/2
        old_cy = y + h/2
        
        # dy, dx: desplazamientos con respecto al centro
        for dy in range(-8, 9, 2):
            for dx in range(-8, 9, 2):
                new_x = int(old_cx - size/2 + dx)
                new_y = int(old_cy - size/2 + dy)
                
                score, box = evaluate_crop_internal(img, new_x, new_y, size, size, session, input_name, output_name)
                if score > best_score:
                    best_score = score
                    best_box = box
                    
    return best_score, best_box

DEFAULT_THRESHOLD = 0.60
DEFAULT_LOW_THRESHOLD = 0.28


def detect_and_draw_eyes(image_path, model_path, threshold=DEFAULT_THRESHOLD, low_threshold=DEFAULT_LOW_THRESHOLD, output_path="./data/resultado_deteccion.jpg"):
    # 1. Verificar existencia del modelo
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo ONNX en: '{model_path}'")
        
    # 2. Cargar el modelo ONNX
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    except Exception as e:
        raise RuntimeError(f"Error al cargar la sesión de ONNX Runtime: {e}")

    # 3. Leer la imagen
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no existe en la ruta: '{image_path}'")
        
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: '{image_path}'")
        
    img_display = img_bgr.copy()
    h_img, w_img, _ = img_bgr.shape
    face_cascade, eye_cascade, eye_glass_cascade, cat_face_cascade = load_cascade_classifiers()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detectores de rostros (humanos y felinos)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
    cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(50, 50))
    
    all_eye_candidates = [] # Lista de (x, y, w, h)
    
    # Buscar ojos dentro de rostros humanos primero si se detectan
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy + int(fh*0.65), fx:fx + fw]
            
            # Detectar con clasificador estándar con parámetros más sensibles dentro del ROI
            eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
                
            # Detectar con clasificador de lentes
            eyes_glass = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes_glass:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))

        if len(all_eye_candidates) == 0:
            eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((ex, ey, ew, eh))
            eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes_glass:
                all_eye_candidates.append((ex, ey, ew, eh))

    # Buscar ojos en rostros de mascotas (gatos/animales)
    if len(cat_faces) > 0:
        for (cx, cy, cw, ch) in cat_faces:
            roi_gray = gray[cy:cy + int(ch*0.60), cx:cx + cw]
            eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.04, minNeighbors=1, minSize=(10, 10))
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((cx + ex, cy + ey, ew, eh))

    # Si no hay caras de ningún tipo, buscar ojos en toda la imagen con clasificadores sensibles
    # Dejamos que la CNN valide con precisión los verdaderos ojos
    if len(all_eye_candidates) == 0 and len(faces) == 0 and len(cat_faces) == 0:
        eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes_std:
            all_eye_candidates.append((ex, ey, ew, eh))
            
        eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes_glass:
            all_eye_candidates.append((ex, ey, ew, eh))
            
    # Eliminar candidatos duplicados o muy solapados (NMS simplificado)
    unique_candidates = []
    for (x, y, w, h) in all_eye_candidates:
        is_dup = False
        for (ux, uy, uw, uh) in unique_candidates:
            center_dist = np.sqrt(((x + w/2) - (ux + uw/2))**2 + ((y + h/2) - (uy + uh/2))**2)
            if center_dist < min(w, uw) * 0.5:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append((x, y, w, h))
    all_eye_candidates = unique_candidates

    # Descartar candidatos fuera de rostros detectados (humano o mascota) si alguno está presente
    has_any_face = (len(faces) > 0 or len(cat_faces) > 0)
    if has_any_face and len(all_eye_candidates) > 0:
        filtered = []
        for (x, y, w, h) in all_eye_candidates:
            cx, cy = x + w / 2, y + h / 2
            inside_face = False
            # Verificar cara humana
            for (fx, fy, fw, fh) in faces:
                if (fx - 15 <= cx <= fx + fw + 15 and fy - 15 <= cy <= fy + int(fh * 0.75)):
                    inside_face = True
                    break
            # Verificar cara felina
            if not inside_face:
                for (cx_f, cy_f, cw_f, ch_f) in cat_faces:
                    if (cx_f - 15 <= cx <= cx_f + cw_f + 15 and cy_f - 15 <= cy <= cy_f + int(ch_f * 0.75)):
                        inside_face = True
                        break
            if inside_face:
                filtered.append((x, y, w, h))
        all_eye_candidates = filtered
            
    # Si no se detectaron candidatos en absoluto, evaluar la imagen en su centro como fallback
    fallback_used = False
    if len(all_eye_candidates) == 0:
        ch = min(224, h_img)
        cw = min(224, w_img)
        cy = (h_img - ch) // 2
        cx = (w_img - cw) // 2
        all_eye_candidates.append((cx, cy, cw, ch))
        fallback_used = True
            
    green_detections = [] # Lista de (x, y, w, h, score) -> score >= threshold
    red_detections = []   # Lista de (x, y, w, h, score) -> low_threshold <= score < threshold
    zero_detections = []  # Lista de (x, y, w, h, score) -> score < low_threshold
    
    print(f"[*] Analizando {len(all_eye_candidates)} candidatos de ojos...")
    
    for (x, y, w, h) in all_eye_candidates:
        if fallback_used:
            score, final_box = evaluate_crop_internal(img_bgr, x, y, w, h, session, input_name, output_name)
            fx, fy, fw, fh = final_box
            if score >= low_threshold:
                red_detections.append((fx, fy, fw, fh, score))
            else:
                zero_detections.append((fx, fy, fw, fh, score))
        else:
            score, final_box = refine_box(img_bgr, x, y, w, h, session, input_name, output_name)
            fx, fy, fw, fh = final_box
            if score >= threshold:
                green_detections.append((fx, fy, fw, fh, score))
            elif score >= low_threshold:
                red_detections.append((fx, fy, fw, fh, score))
            else:
                zero_detections.append((fx, fy, fw, fh, score))
            
    # 5. Dibujar los resultados según las especificaciones
    print("-" * 60)
    print(" RESULTADO DE DETECCIÓN ")
    print("-" * 60)
    
    has_boxes = (len(green_detections) > 0 or len(red_detections) > 0)
    
    if has_boxes:
        # Se detectaron coincidencias por encima de cero (>= low_threshold)
        print(f"[+] Detección con coincidencia.")
        
        # Procesar primero las rojas (coincidencia baja/media)
        for i, (x, y, w, h, score) in enumerate(red_detections):
            score_text = f"{score:.4f}"
            print(f"  [Coincidencia Baja (Cuadro Rojo)] Box: [{x}, {y}, {w}, {h}] | Confianza: {score_text}")
            
        # Procesar las verdes (coincidencia alta, >= threshold)
        for i, (x, y, w, h, score) in enumerate(green_detections):
            score_text = f"{score:.4f}"
            print(f"  [Ojo Detectado (Cuadro Verde)] Box: [{x}, {y}, {w}, {h}] | Confianza: {score_text}")
            
    else:
        # No se detectó ningún ojo ni coincidencia válida (todos < low_threshold)
        # Tomar la puntuación máxima obtenida entre las evaluadas
        all_evaluated = green_detections + red_detections + zero_detections
        max_score = max(item[4] for item in all_evaluated) if len(all_evaluated) > 0 else 0.0000
        
        print(f"[!] No se detectaron ojos. Puntuación máxima obtenida (cercana a cero): {max_score:.6f}")
        print("[*] Modo fallback o scores muy bajos...")
        
    print("-" * 60)
    
    # 6. Guardar imagen final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_display)
    print(f"[+] Imagen resultante guardada exitosamente en: '{output_path}'")
    
    return len(green_detections) + len(red_detections), not has_boxes

def main():
    parser = argparse.ArgumentParser(
        description="Script de Detección de Ojos utilizando Haar Cascades e inferencia ONNX para validación y fallback de visualización.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Ruta a la imagen de entrada local."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="./model/tu_modelo.onnx",
        help="Ruta al modelo ONNX entrenado."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Umbral de validación para caja verde del clasificador ONNX."
    )
    parser.add_argument(
        "-lt", "--low-threshold",
        type=float,
        default=DEFAULT_LOW_THRESHOLD,
        help="Umbral inferior para marcar coincidencia sospechosa con caja roja."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/resultado_deteccion.jpg",
        help="Ruta de salida para guardar la imagen analizada con las cajas."
    )
    
    args = parser.parse_args()
    
    try:
        count, fallback = detect_and_draw_eyes(
            image_path=args.image,
            model_path=args.model,
            threshold=args.threshold,
            low_threshold=args.low_threshold,
            output_path=args.output
        )
        print(f"\n[ÉXITO] Detección finalizada. Ojos procesados: {count} | Modo Fallback: {fallback}")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error al ejecutar la detección: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
