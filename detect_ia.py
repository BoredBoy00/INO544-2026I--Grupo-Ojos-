import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime as ort

def preprocess_image(image_path, target_size=(224, 224), normalize=True):
    """
    Lee y preprocesa una imagen para la inferencia con el modelo CNN en formato ONNX.
    
    Pasos:
    1. Carga la imagen local en formato BGR.
    2. Convierte de BGR a RGB.
    3. Redimensiona exactamente a target_size (224x224).
    4. Normaliza los píxeles a float32 en rango [0.0, 1.0] (si normalize es True).
    5. Añade la dimensión de lote (Batch dimension) resultando en shape [1, H, W, C].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no existe en la ruta especificada: '{image_path}'")

    # 1. Leer la imagen local en formato BGR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo decodificar la imagen. El archivo podría estar corrupto o no ser una imagen válida: '{image_path}'")

    # 2. Convertir a RGB (estándar para la mayoría de CNNs)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Redimensionar exactamente a 224x224 píxeles
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # 4. Convertir a float32 y normalizar
    img_data = img_resized.astype(np.float32)
    if normalize:
        # División explícita y parametrizable entre 255.0 para el rango [0.0, 1.0]
        img_data /= 255.0

    # 5. Añadir dimensión de lote (Batch) para obtener la forma [1, 224, 224, 3]
    img_tensor = np.expand_dims(img_data, axis=0)
    
    return img_tensor

def main():
    # Configuración de los argumentos por línea de comandos
    parser = argparse.ArgumentParser(
        description="Script de detección de IA en imágenes utilizando un modelo entrenado en formato ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Ruta de la imagen local que se desea analizar."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="./model/tu_modelo.onnx",
        help="Ruta del archivo de modelo ONNX."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Umbral de confianza para clasificar la imagen como IA (si es mayor que el umbral)."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Desactiva la normalización de píxeles (división por 255.0)."
    )
    
    args = parser.parse_args()

    print("=" * 60)
    print(" DETECTOR DE IMÁGENES GENERADAS POR IA - INFERENCIA ONNX ")
    print("=" * 60)

    # --- 1. Cargar el Modelo ONNX ---
    print(f"[*] Cargando modelo desde: {args.model}")
    if not os.path.exists(args.model):
        print(f"[ERROR] No se encontró el archivo del modelo ONNX en: '{args.model}'")
        print("        Por favor, asegúrate de colocar el archivo del modelo entrenado en esa ruta")
        print("        o usa el script de prueba para generar un modelo dummy.")
        sys.exit(1)

    try:
        # Inicializar la sesión de ONNX Runtime
        # Se prioriza la compatibilidad offline con CPU, pero se autodetecta el hardware
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        
        # Obtener nombres y detalles de entrada/salida para validación
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        
        input_name = input_meta.name
        output_name = output_meta.name
        
        print(f"[+] Modelo cargado exitosamente.")
        print(f"    - Capa de entrada (Input Name): '{input_name}' | Shape: {input_meta.shape} | Tipo: {input_meta.type}")
        print(f"    - Capa de salida (Output Name): '{output_name}' | Shape: {output_meta.shape} | Tipo: {output_meta.type}")
        
        # Validar especificaciones requeridas
        if input_name != "cam_input":
            print(f"[WARNING] El nombre de la entrada del modelo es '{input_name}', pero se esperaba 'cam_input'.")
        if output_name != "confidence_score":
            print(f"[WARNING] El nombre de la salida del modelo es '{output_name}', pero se esperaba 'confidence_score'.")

    except Exception as e:
        print(f"[ERROR] Error crítico al cargar o validar el modelo ONNX: {e}")
        sys.exit(1)

    # --- 2. Leer y Preprocesar la Imagen ---
    print(f"[*] Leyendo y preprocesando la imagen: {args.image}")
    try:
        input_tensor = preprocess_image(
            image_path=args.image,
            target_size=(224, 224),
            normalize=args.normalize
        )
        print(f"[+] Preprocesamiento completado con éxito.")
        print(f"    - Forma del tensor de entrada (Shape): {input_tensor.shape}")
        print(f"    - Rango de valores: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error inesperado durante el preprocesamiento de la imagen: {e}")
        sys.exit(1)

    # --- 3. Ejecutar la Inferencia ---
    print("[*] Ejecutando inferencia en el modelo...")
    try:
        # Ejecutar el modelo alimentando la capa 'cam_input' con nuestro tensor preprocesado
        outputs = session.run([output_name], {input_name: input_tensor})
        
        # La salida del modelo es confidence_score de forma [1, 1]
        score = float(outputs[0][0][0])
        
        # --- 4. Determinar Veredicto y Mostrar Resultados ---
        # Si score > threshold, se considera "IA", de lo contrario "Real"
        veredicto = "IA" if score > args.threshold else "Real"
        
        print("-" * 60)
        print(" RESULTADO DEL ANÁLISIS ")
        print("-" * 60)
        print(f"  > Estado de la lectura:  EXITOSO")
        print(f"  > Puntuación de confianza (Score): {score:.4f} ({score * 100:.2f}%)")
        print(f"  > Umbral configurado:   {args.threshold:.2f}")
        
        if veredicto == "IA":
            print(f"  > VEREDICTO FINAL:      [ IA ] (La imagen muestra indicios de haber sido generada por IA)")
        else:
            print(f"  > VEREDICTO FINAL:      [ REAL ] (La imagen parece ser una fotografía o creación real)")
        print("-" * 60)

    except Exception as e:
        print(f"[ERROR] Ocurrió un error al ejecutar la inferencia del modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
