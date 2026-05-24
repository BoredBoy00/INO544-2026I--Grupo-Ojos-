import os
import cv2
import numpy as np

def create_test_image(output_path="./data/imagen_prueba.jpg"):
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Crear una imagen aleatoria de 400x400 con 3 canales (BGR)
    # Rango [0, 255] uint8
    img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    
    # Dibujar un círculo simple para que tenga algo de contenido
    cv2.circle(img, (200, 200), 100, (0, 255, 0), -1) # Círculo verde
    
    # Guardar la imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen de prueba generada exitosamente en: {output_path}")

if __name__ == "__main__":
    create_test_image()
