import os
import cv2
import numpy as np

def main():
    neg_dir = "./Ojos_negativos"
    os.makedirs(neg_dir, exist_ok=True)
    
    # Generador de números pseudoaleatorios con semilla para reproducibilidad
    rng = np.random.default_rng(42)
    
    # Generar 50 imágenes sintéticas de formas circulares no orgánicas (falsos ojos típicos)
    for i in range(50):
        # Lienzo en blanco con colores variados
        bg_color = rng.integers(50, 200, size=3).tolist()
        img = np.full((120, 120, 3), bg_color, dtype=np.uint8)
        
        # Centro de la figura
        cx, cy = 60, 60
        
        if i % 4 == 0:
            # Caso 1: Lupa digital (Círculo metálico brillante con mango y destello blanco de cristal)
            r = int(rng.integers(25, 40))
            thickness = int(rng.integers(3, 7))
            # Dibujar aro exterior (metalizado)
            cv2.circle(img, (cx, cy), r, (180, 180, 180), thickness)
            # Dibujar cristal interior (azulino/transparente)
            cv2.circle(img, (cx, cy), r - thickness, (220, 240, 255), -1)
            # Dibujar reflejo o destello de cristal (círculo blanco excéntrico de alto brillo)
            cv2.circle(img, (cx - 10, cy - 10), int(rng.integers(3, 7)), (255, 255, 255), -1)
            # Dibujar mango de la lupa
            cv2.line(img, (cx + r - 3, cy + r - 3), (cx + r + 20, cy + r + 20), (100, 100, 100), thickness + 2)
            
        elif i % 4 == 1:
            # Caso 2: Botón de ropa (Aro de color plástico con 4 agujeros oscuros perfectos)
            r = int(rng.integers(30, 45))
            color = rng.integers(0, 120, size=3).tolist()
            # Borde y cara del botón
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
            # 4 Agujeros concéntricos oscuros
            hole_r = int(rng.integers(3, 5))
            cv2.circle(img, (cx - 10, cy - 10), hole_r, (0, 0, 0), -1)
            cv2.circle(img, (cx + 10, cy - 10), hole_r, (0, 0, 0), -1)
            cv2.circle(img, (cx - 10, cy + 10), hole_r, (0, 0, 0), -1)
            cv2.circle(img, (cx + 10, cy + 10), hole_r, (0, 0, 0), -1)
            
        elif i % 4 == 2:
            # Caso 3: Círculos concéntricos perfectos (Típico diana, plato o moneda circular inorgánica)
            r_max = int(rng.integers(35, 50))
            color1 = rng.integers(0, 150, size=3).tolist()
            color2 = rng.integers(0, 150, size=3).tolist()
            cv2.circle(img, (cx, cy), r_max, color1, 2)
            cv2.circle(img, (cx, cy), r_max - 10, color2, 2)
            cv2.circle(img, (cx, cy), r_max - 20, (0, 0, 0), -1)
            
        else:
            # Caso 4: Bokeh / Destellos circulares de lentes o luces artificiales
            color = rng.integers(200, 256, size=3).tolist()
            cv2.circle(img, (cx, cy), int(rng.integers(15, 30)), color, -1)
            # Desenfoque Gaussiano severo para imitar desenfoque de cámara
            img = cv2.GaussianBlur(img, (15, 15), 0)
            
        # Añadir ruido Gaussiano de fondo para aumentar realismo y obligar a la CNN a filtrar ruido
        noise = rng.integers(-15, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        path = os.path.join(neg_dir, f"synth_hard_neg_{i}.jpg")
        cv2.imwrite(path, img)
        
    print(f"[+] Generados 50 negativos difíciles sintéticos en '{neg_dir}' exitosamente.")

if __name__ == "__main__":
    main()
