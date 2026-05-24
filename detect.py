import cv2
import numpy as np
import os

# Ajustes de dimensiones acordes a la pizarra
IMG_SIZE = (224, 224)

def run_inference_onnx(image_path, model_path="eye_detector.onnx", threshold=0.85):
    """
    Realiza la detección de ojos en una imagen estática usando el modelo ONNX exportado.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] Error: Se requiere 'onnxruntime' para cargar el modelo ONNX.")
        print("    Instálalo usando: python -m pip install onnxruntime")
        return

    if not os.path.exists(model_path):
        print(f"[!] No se encontró el modelo ONNX en '{model_path}'.")
        print("    Asegúrate de entrenar y exportar el modelo primero ejecutando 'datasep.py'.")
        return

    # 1. Cargar sesión de inferencia de ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name     # "cam_input"
    output_name = session.get_outputs()[0].name   # "confidence_score"

    # 2. Cargar clasificadores en cascada de Haar de OpenCV para detección facial básica
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] No se pudo cargar la imagen en '{image_path}'")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    print(f"[*] Rostros detectados: {len(faces)}")
    output_img = img.copy()

    for (fx, fy, fw, fh) in faces:
        # Dibujar recuadro de rostro en azul
        cv2.rectangle(output_img, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        
        # Explorar la mitad superior del rostro (zona típica de ojos)
        eye_region_height = int(fh * 0.55)
        eye_region = img[fy:fy + eye_region_height, fx:fx+fw]

        # Ventana deslizante para escaneo de ojos (ventana de 64x64 deslizada de 8 en 8 píxeles)
        step, win = 8, 64
        for y in range(0, eye_region.shape[0] - win, step):
            for x in range(0, eye_region.shape[1] - win, step):
                patch = eye_region[y:y+win, x:x+win]
                if patch.shape[0] != win or patch.shape[1] != win:
                    continue

                # Preprocesamiento idéntico al entrenamiento:
                # BGR a RGB
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                # Redimensionar a 224x224x3 (como requiere el modelo)
                patch_resized = cv2.resize(patch_rgb, IMG_SIZE)
                # Normalizar a rango [0.0, 1.0] (float32)
                patch_normalized = patch_resized.astype(np.float32) / 255.0
                # Añadir dimensión de batch: [1, 224, 224, 3] (NHWC)
                patch_input = np.expand_dims(patch_normalized, axis=0)

                # Ejecutar inferencia en ONNX Runtime
                outputs = session.run([output_name], {input_name: patch_input})
                prob = float(outputs[0][0][0])  # Confianza entregada por la Sigmoid

                if prob > threshold:
                    # Coordenadas relativas a la imagen original
                    ox, oy = fx + x, fy + y
                    # Dibujar recuadro de ojo en verde
                    cv2.rectangle(output_img, (ox, oy), (ox+win, oy+win), (0, 255, 0), 1)
                    # Añadir la confianza en texto pequeño
                    cv2.putText(output_img, f"{prob:.1%}", (ox, oy - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Guardar imagen resultante
    result_path = "resultado_deteccion.jpg"
    cv2.imwrite(result_path, output_img)
    print(f"[+] Detección completada. Imagen guardada en: '{result_path}'")

def run_webcam_onnx(model_path="eye_detector.onnx", threshold=0.85):
    """
    Ejecuta la detección de ojos en tiempo real usando tu cámara web y el modelo ONNX.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] Error: Se requiere 'onnxruntime' para cargar el modelo ONNX.")
        return

    if not os.path.exists(model_path):
        print(f"[!] No se encontró el modelo ONNX en '{model_path}'. Entrena primero.")
        return

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] No se pudo abrir la cámara web.")
        return

    print("[*] Iniciando cámara web... Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            
            eye_region = frame[fy:fy + int(fh*0.55), fx:fx+fw]
            step, win = 16, 64  # Aumentamos el paso (step) a 16 para mayor velocidad en tiempo real
            
            for y in range(0, eye_region.shape[0] - win, step):
                for x in range(0, eye_region.shape[1] - win, step):
                    patch = eye_region[y:y+win, x:x+win]
                    if patch.shape[0] != win or patch.shape[1] != win:
                        continue

                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    patch_resized = cv2.resize(patch_rgb, IMG_SIZE)
                    patch_normalized = patch_resized.astype(np.float32) / 255.0
                    patch_input = np.expand_dims(patch_normalized, axis=0)

                    outputs = session.run([output_name], {input_name: patch_input})
                    prob = float(outputs[0][0][0])

                    if prob > threshold:
                        ox, oy = fx + x, fy + y
                        cv2.rectangle(frame, (ox, oy), (ox+win, oy+win), (0, 255, 0), 2)
                        cv2.putText(frame, f"Ojo: {prob:.1%}", (ox, oy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Detector de Ojos (ONNX)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script para utilizar el modelo ONNX exportado")
    parser.add_argument("--image", type=str, default=None, help="Ruta de una imagen para detectar ojos")
    parser.add_argument("--webcam", action="store_true", help="Iniciar detección usando la cámara web")
    
    args = parser.parse_args()
    
    if args.image:
        run_inference_onnx(args.image)
    elif args.webcam:
        run_webcam_onnx()
    else:
        print("[!] Por favor especifica una acción:")
        print("    - Para imagen:  python detect.py --image RUTA_A_TU_IMAGEN.jpg")
        print("    - Para cámara:  python detect.py --webcam")
