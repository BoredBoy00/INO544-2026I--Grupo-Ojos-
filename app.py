import os
import cv2
import numpy as np
import base64
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

# Inicializar aplicación Flask
app = Flask(__name__, static_folder="static", template_folder="templates")

# Cargar clasificadores Haar Cascade de OpenCV
try:
    face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
    
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)
    print("[*] Clasificadores Haar Cascade cargados correctamente.")
except Exception as e:
    print(f"[!] Error al cargar clasificadores Haar Cascade: {e}")

# Cargar el modelo ONNX globalmente para evitar latencias en la inferencia
MODEL_PATH = "./model/tu_modelo.onnx"
session = None
input_name = None
output_name = None

try:
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"[+] Modelo ONNX cargado globalmente desde '{MODEL_PATH}'.")
    else:
        print(f"[WARNING] No se encontró el modelo ONNX en '{MODEL_PATH}'. Ejecute primero train_model.py.")
except Exception as e:
    print(f"[!] Error al cargar el modelo ONNX: {e}")


def evaluate_crop_internal(img, x, y, w, h):
    """
    Extrae, redimensiona y evalúa un parche de imagen utilizando el modelo ONNX global.
    """
    global session, input_name, output_name
    if session is None:
        return 0.0, (x, y, w, h)
        
    img_h, img_w, _ = img.shape
    x1 = int(max(0, min(x, img_w - 10)))
    y1 = int(max(0, min(y, img_h - 10)))
    w1 = int(max(10, min(w, img_w - x1)))
    h1 = int(max(10, min(h, img_h - y1)))
    
    crop = img[y1:y1+h1, x1:x1+w1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224, 224))
    
    input_tensor = crop_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    outputs = session.run([output_name], {input_name: input_tensor})
    score = float(outputs[0][0][0])
    return score, (x1, y1, w1, h1)


def refine_box(img, x, y, w, h):
    """
    Realiza una búsqueda local alrededor del candidato original para maximizar la puntuación ONNX.
    """
    best_score, best_box = evaluate_crop_internal(img, x, y, w, h)
    
    # Búsqueda local de cuadrícula
    for ds in range(-6, 7, 2):
        size = w + ds
        if size < 15:
            continue
        old_cx = x + w/2
        old_cy = y + h/2
        
        for dy in range(-8, 9, 2):
            for dx in range(-8, 9, 2):
                new_x = int(old_cx - size/2 + dx)
                new_y = int(old_cy - size/2 + dy)
                
                score, box = evaluate_crop_internal(img, new_x, new_y, size, size)
                if score > best_score:
                    best_score = score
                    best_box = box
                    
    return best_score, best_box


@app.route("/")
def index():
    """Sirve la página web principal del dashboard."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    Endpoint para procesar la imagen enviada por el cliente.
    Acepta formato JSON con la imagen codificada en Base64, o carga de archivos multipart.
    """
    try:
        # 1. Obtener los umbrales configurables
        threshold = float(request.values.get("threshold", 0.5))
        low_threshold = float(request.values.get("low_threshold", 0.001))
        
        img_bgr = None
        
        # Verificar si es una carga de archivo normal
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
        # O si es JSON con base64 (usado por la captura de la cámara en vivo)
        elif request.is_json:
            data = request.get_json()
            image_data = data.get("image", "")
            if image_data:
                # Quitar prefijo de datos si está presente (e.g., "data:image/jpeg;base64,")
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                threshold = float(data.get("threshold", threshold))
                low_threshold = float(data.get("low_threshold", low_threshold))
        
        if img_bgr is None:
            return jsonify({"success": False, "error": "No se recibió ninguna imagen válida."}), 400
            
        h_img, w_img, _ = img_bgr.shape
        img_display = img_bgr.copy()
        
        # Convertir a escala de grises para los clasificadores en cascada
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detector de rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
        
        all_eye_candidates = []
        
        # Buscar ojos dentro de rostros
        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                roi_gray = gray[fy:fy + int(fh*0.65), fx:fx + fw]
                
                eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes_std:
                    all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
                    
                eyes_glass = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes_glass:
                    all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
        else:
            # Si no hay caras, buscar en toda la imagen con parámetros más estrictos
            eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((ex, ey, ew, eh))
                
            eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
            for (ex, ey, ew, eh) in eyes_glass:
                all_eye_candidates.append((ex, ey, ew, eh))
                
        # NMS Simplificado para eliminar duplicados
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
        
        # Fallback al centro si no hay candidatos en absoluto
        fallback_used = False
        if len(all_eye_candidates) == 0:
            ch = min(224, h_img)
            cw = min(224, w_img)
            cy = (h_img - ch) // 2
            cx = (w_img - cw) // 2
            all_eye_candidates.append((cx, cy, cw, ch))
            fallback_used = True
            
        green_detections = [] # (x, y, w, h, score)
        red_detections = []   # (x, y, w, h, score)
        zero_detections = []  # (x, y, w, h, score)
        
        for (x, y, w, h) in all_eye_candidates:
            if fallback_used:
                score, final_box = evaluate_crop_internal(img_bgr, x, y, w, h)
            else:
                score, final_box = refine_box(img_bgr, x, y, w, h)
                
            fx, fy, fw, fh = final_box
            if score >= threshold:
                green_detections.append((fx, fy, fw, fh, score))
            elif score >= low_threshold:
                red_detections.append((fx, fy, fw, fh, score))
            else:
                zero_detections.append((fx, fy, fw, fh, score))
                    
        has_boxes = (len(green_detections) > 0 or len(red_detections) > 0)
        detections_list = []
        
        if has_boxes:
            # Dibujar cajas rojas (coincidencia baja/media)
            for (x, y, w, h, score) in red_detections:
                color = (0, 0, 255) # Rojo en BGR
                cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
                
                score_text = f"{score:.4f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                ty = y - 8 if y - 8 > 20 else y + h + 20
                cv2.putText(img_display, score_text, (x, ty), font, 0.6, color, 1, cv2.LINE_AA)
                
                detections_list.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "score": round(score, 4),
                    "color": "red",
                    "class": "Coincidencia Baja"
                })
                
            # Dibujar cajas verdes (ojo detectado, >= threshold)
            for (x, y, w, h, score) in green_detections:
                color = (0, 255, 0) # Verde en BGR
                cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
                
                score_text = f"{score:.4f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                ty = y - 8 if y - 8 > 20 else y + h + 20
                cv2.putText(img_display, score_text, (x, ty), font, 0.6, color, 1, cv2.LINE_AA)
                
                detections_list.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "score": round(score, 4),
                    "color": "green",
                    "class": "Ojo Detectado"
                })
                
            max_score = max([d["score"] for d in detections_list])
            verdict = "Ojo Detectado" if len(green_detections) > 0 else "Coincidencia Baja"
            
        else:
            # No se detectó ojo, mostrar puntuación en rojo en el centro
            all_evaluated = green_detections + red_detections + zero_detections
            max_score = max(item[4] for item in all_evaluated) if len(all_evaluated) > 0 else 0.0
            
            score_text = f"{max_score:.4f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(score_text, font, 1.0, 2)
            tx = (w_img - text_size[0]) // 2
            ty = (h_img + text_size[1]) // 2
            color = (0, 0, 255) # Rojo en BGR
            cv2.putText(img_display, score_text, (tx, ty), font, 1.0, color, 2, cv2.LINE_AA)
            
            verdict = "No se detectaron ojos"
            detections_list.append({
                "box": [0, 0, 0, 0] if (fallback_used or not all_evaluated) else [int(all_evaluated[0][0]), int(all_evaluated[0][1]), int(all_evaluated[0][2]), int(all_evaluated[0][3])],
                "score": round(max_score, 4),
                "color": "none",
                "class": "No es Ojo"
            })
            
        # Codificar la imagen procesada a Base64 para enviarla al cliente
        _, buffer = cv2.imencode(".jpg", img_display)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        response_data = {
            "success": True,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections_list,
            "max_score": round(max_score, 4),
            "has_detections": has_boxes,
            "verdict": verdict,
            "fallback_used": fallback_used
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[!] Error durante el procesamiento de la imagen: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Configurar puerto predeterminado 5000 y habilitar escucha en red local
    # Se puede cambiar host a "127.0.0.1" si se prefiere restringir a local
    app.run(host="0.0.0.0", port=5000, debug=True)
