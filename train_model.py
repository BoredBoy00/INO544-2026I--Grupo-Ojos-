import sys
import io
# Evitar errores de codificación de caracteres especiales/emojis en la consola de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Dimensiones acordes a la pizarra
INFER_SIZE = 224
RNG = np.random.default_rng(42)

# --- ARQUITECTURA DE LA CNN EN PYTORCH ---
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        # Entrada del modelo ONNX (cam_input): [1, 224, 224, 3] (NHWC)
        # Reducción/Submuestreo interno adaptativo a 64x64 para mejor retención de detalles finos
        self.downsample = nn.AdaptiveAvgPool2d((64, 64))
        
        # Capa Convolucional 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: [Batch, 16, 32, 32]
        
        # Capa Convolucional 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: [Batch, 32, 16, 16]
        
        # Capa Convolucional 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: [Batch, 64, 8, 8]
        
        # Capa de regularización
        self.dropout = nn.Dropout(0.3)
        
        # Capa Totalmente Conectada (Clasificador lineal)
        self.fc = nn.Linear(64 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, 224, 224, 3]
        # Transponer dimensiones de NHWC a NCHW para PyTorch: [Batch, 3, 224, 224]
        x = x.permute(0, 3, 1, 2)
        # Submuestrear
        x = self.downsample(x)
        # Convoluciones con Normalización de Lote
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        # Aplanar
        x = x.reshape(x.size(0), -1)
        # Regularización y clasificador
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def is_overlapping(box1, box2, margin=10):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x2_m = max(0, x2 - margin)
    y2_m = max(0, y2 - margin)
    w2_m = w2 + 2 * margin
    h2_m = h2 + 2 * margin
    if x1 + w1 <= x2_m or x2_m + w2_m <= x1:
        return False
    if y1 + h1 <= y2_m or y2_m + h2_m <= y1:
        return False
    return True


def clamp_box(x, y, w, h, img_w, img_h):
    x = max(0, min(int(x), img_w - 10))
    y = max(0, min(int(y), img_h - 10))
    w = max(10, min(int(w), img_w - x))
    h = max(10, min(int(h), img_h - y))
    return x, y, w, h


def apply_clahe_color(crop_bgr):
    """
    Aplica CLAHE al canal de luminosidad (L) en el espacio de color LAB
    para mejorar el contraste adaptativo sin distorsionar los colores del parche.
    """
    if crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return crop_bgr
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced_bgr
    except Exception:
        return crop_bgr


def crop_bgr_to_rgb(img_bgr, x, y, w, h):
    x, y, w, h = clamp_box(x, y, w, h, img_bgr.shape[1], img_bgr.shape[0])
    crop = img_bgr[y : y + h, x : x + w]
    crop_enhanced = apply_clahe_color(crop)
    return cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)


def patch_to_features_cnn(crop_rgb):
    # Redimensionar a [224, 224, 3] en RGB y normalizar a [0.0, 1.0]
    resized_rgb = cv2.resize(crop_rgb, (INFER_SIZE, INFER_SIZE)).astype(np.float32) / 255.0
    return resized_rgb


def load_extra_negatives(neg_dir="./Ojos_negativos"):
    """Parches de escenas sin ojo: centro, cuadrantes y bandas (ropa, fondos)."""
    if not os.path.isdir(neg_dir):
        return []
    paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(glob.glob(os.path.join(neg_dir, pattern)))
    crops = []
    for path in sorted(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        ch = min(INFER_SIZE, h)
        cw = min(INFER_SIZE, w)
        cy = (h - ch) // 2
        cx = (w - cw) // 2
        patches = [
            (cx, cy, cw, ch),
            (0, 0, cw, ch),
            (max(0, w - cw), 0, cw, ch),
            (0, max(0, h - ch), cw, ch),
            (max(0, w - cw), max(0, h - ch), cw, ch),
            (cx, int(h * 0.45), cw, ch),
            (cx, int(h * 0.55), cw, ch),
        ]
        for (px, py, pw, ph) in patches:
            crops.append(crop_bgr_to_rgb(img, px, py, pw, ph))
        # Peso extra: repetir parches de carpetas de hard negatives
        for _ in range(4):
            crops.extend(crops[-len(patches) :])
    return crops


def load_dataset(ojos_dir="./Ojos"):
    print("[*] Cargando imágenes de la carpeta Ojos...")
    image_paths = sorted(glob.glob(os.path.join(ojos_dir, "img_transformada_*.jpg")))
    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes en '{ojos_dir}' con el patrón 'img_transformada_*.jpg'"
        )

    eye_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    )

    pos_crops = []
    neg_crops = []
    stats = {
        "total": len(image_paths),
        "haar_ok": 0,
        "haar_fail_skipped": 0,
        "multi_eye_used": 0,
    }

    print("[*] Extrayendo parches (solo positivos con Haar confirmado)...")
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_eyes = eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(detected_eyes) == 0:
            stats["haar_fail_skipped"] += 1
            continue

        stats["haar_ok"] += 1
        eye_boxes = [list(box) for box in detected_eyes]

        for (ex, ey, ew, eh) in detected_eyes[:2]:
            crop_rgb = crop_bgr_to_rgb(img, ex, ey, ew, eh)
            pos_crops.append(crop_rgb)
            
            # --- AUMENTO DE DATOS DINÁMICO ---
            # 1. Volteo Horizontal (Simula ojos del lado opuesto)
            flipped = cv2.flip(crop_rgb, 1)
            pos_crops.append(flipped)
            
            # 2. Ajuste leve de brillo y contraste dinámico (Simula variabilidad lumínica)
            alpha = RNG.uniform(0.85, 1.15)
            beta = RNG.uniform(-10, 10)
            adjusted = cv2.convertScaleAbs(crop_rgb, alpha=alpha, beta=beta)
            pos_crops.append(adjusted)
            
            if len(detected_eyes) > 1:
                stats["multi_eye_used"] += 1

        # Negativos aleatorios sin solapar con ojos
        for _ in range(2):
            for _attempt in range(120):
                nw = int(RNG.integers(40, min(90, w)))
                nh = nw
                if nw >= w or nh >= h:
                    continue
                nx = int(RNG.integers(0, w - nw))
                ny = int(RNG.integers(0, h - nh))
                neg_box = (nx, ny, nw, nh)
                if not any(is_overlapping(neg_box, eb) for eb in eye_boxes):
                    neg_crops.append(crop_bgr_to_rgb(img, nx, ny, nw, nh))
                    break

        # Negativo: recorte central (simula modo fallback de la cámara)
        ch = min(INFER_SIZE, h)
        cw = min(INFER_SIZE, w)
        cy = (h - ch) // 2
        cx = (w - cw) // 2
        center_box = (cx, cy, cw, ch)
        if not any(is_overlapping(center_box, eb, margin=20) for eb in eye_boxes):
            neg_crops.append(crop_bgr_to_rgb(img, cx, cy, cw, ch))

        # Negativo: tercio inferior (torso / ropa / texturas)
        ly = int(h * 0.55)
        lh = h - ly
        if lh >= 40:
            lw = min(int(w * 0.7), w)
            lx = (w - lw) // 2
            lower_box = (lx, ly, lw, lh)
            if not any(is_overlapping(lower_box, eb, margin=15) for eb in eye_boxes):
                neg_crops.append(crop_bgr_to_rgb(img, lx, ly, lw, lh))

    extra_neg = load_extra_negatives()
    if extra_neg:
        neg_crops.extend(extra_neg)
        print(f"[+] Añadidos {len(extra_neg)} parches desde ./Ojos_negativos/")

    # Negativos sintéticos (ruido y bloques mezclados)
    print("[*] Generando muestras negativas sintéticas...")
    synth_count = min(len(pos_crops), 400)
    for i in range(synth_count):
        if i % 2 == 0:
            noise = RNG.integers(0, 256, (60, 60, 3), dtype=np.uint8)
            neg_crops.append(noise)
        else:
            base = pos_crops[i % len(pos_crops)].copy()
            bh, bw = base.shape[:2]
            block_size = max(4, bh // 4)
            blocks = []
            for r in range(0, bh, block_size):
                for col in range(0, bw, block_size):
                    b = base[r : r + block_size, col : col + block_size]
                    if b.size > 0:
                        blocks.append(b)
            RNG.shuffle(blocks)
            scrambled = np.zeros_like(base)
            idx = 0
            for r in range(0, bh, block_size):
                for col in range(0, bw, block_size):
                    if idx < len(blocks):
                        target = scrambled[r : r + block_size, col : col + block_size]
                        if target.size > 0:
                            scrambled[r : r + block_size, col : col + block_size] = cv2.resize(
                                blocks[idx], (target.shape[1], target.shape[0])
                            )
                        idx += 1
            neg_crops.append(scrambled)

    # --- AUMENTO DE DATOS DINÁMICO EN NEGATIVOS ---
    augmented_negatives = []
    for crop in neg_crops:
        # Añadir original
        augmented_negatives.append(crop)
        
        # 1. Volteo Horizontal (Simula simetría)
        flipped = cv2.flip(crop, 1)
        augmented_negatives.append(flipped)
        
        # 2. Ajuste leve de brillo y contraste dinámico
        alpha = RNG.uniform(0.85, 1.15)
        beta = RNG.uniform(-10, 10)
        adjusted = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        augmented_negatives.append(adjusted)
        
    neg_crops = augmented_negatives

    pos_count = len(pos_crops)
    neg_count = len(neg_crops)
    print("\n[*] --- Auditoría del dataset ---")
    print(f"    Imágenes en carpeta:        {stats['total']}")
    print(f"    Con ojo detectado (Haar):   {stats['haar_ok']}")
    print(f"    Omitidas (sin Haar):        {stats['haar_fail_skipped']}")
    print(f"    Parches positivos:          {pos_count}")
    print(f"    Parches negativos:          {neg_count}")
    if stats["haar_fail_skipped"] > 0:
        pct = 100.0 * stats["haar_fail_skipped"] / max(1, stats["total"])
        print(f"    [!] {pct:.1f}% de imágenes sin ojo Haar — revisar calidad o iluminación.")

    if pos_count < 50:
        raise ValueError(
            f"Muy pocos positivos ({pos_count}). Añade más imágenes con ojos visibles en ./Ojos/"
        )

    # Extraer parches de imagen y convertirlos en arrays RGB de 224x224x3 normalizados
    print("[*] Redimensionando y preprocesando parches a 224x224x3...")
    X_pos = np.array([patch_to_features_cnn(c) for c in pos_crops], dtype=np.float32)
    X_neg = np.array([patch_to_features_cnn(c) for c in neg_crops], dtype=np.float32)

    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.vstack(
        (
            np.ones((pos_count, 1), dtype=np.float32),
            np.zeros((neg_count, 1), dtype=np.float32),
        )
    )

    return X, y


def train_cnn(X, y, epochs=15, lr=0.001, batch_size=32):
    # Mezclar el dataset completo antes de separar
    indices = np.arange(X.shape[0])
    RNG.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # --- IMPLEMENTACIÓN DEL SPLIT 80/20 TRAIN/TEST ---
    split_idx = int(0.8 * X.shape[0])
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\n[*] --- Configuración del Split 80/20 ---")
    print(f"    Total de muestras:    {X.shape[0]}")
    print(f"    Muestras Entrenamiento (80%): {X_train.shape[0]}")
    print(f"    Muestras Validación (20%):   {X_val.shape[0]}")

    model = EyeCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_samples = X_train.shape[0]

    print("\n[*] Entrenando la Red Neuronal Convolucional (CNN)...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Mezclar para cada época
        perm = torch.randperm(num_samples)
        X_train_t = torch.tensor(X_train[perm])
        y_train_t = torch.tensor(y_train[perm])
        
        for i in range(0, num_samples, batch_size):
            x_batch = X_train_t[i : i + batch_size]
            y_batch = y_train_t[i : i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)
            
        train_loss = epoch_loss / num_samples
        train_acc = (correct / total) * 100.0
        
        # Evaluación en conjunto de validación
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val)
            y_val_t = torch.tensor(y_val)
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_preds = (val_outputs > 0.5).float()
            val_correct = (val_preds == y_val_t).sum().item()
            val_acc = (val_correct / X_val_t.size(0)) * 100.0
            
        print(
            f"    Época {epoch:02d}/{epochs:02d} | "
            f"Loss Train: {train_loss:.5f} | Acc Train: {train_acc:.2f}% | "
            f"Loss Val: {val_loss:.5f} | Acc Val: {val_acc:.2f}%"
        )

    print("[+] Entrenamiento de CNN completado.")
    return model


def export_to_onnx(model, output_path="./model/tu_modelo.onnx"):
    print(f"\n[*] Exportando modelo CNN a formato ONNX...")
    model.eval()
    dummy_input = torch.zeros(1, INFER_SIZE, INFER_SIZE, 3, dtype=torch.float32)

    # 1. Obtener la ruta del directorio destino y el nombre del archivo
    final_onnx_path = os.path.abspath(output_path)
    dest_dir = os.path.dirname(final_onnx_path)
    filename = os.path.basename(final_onnx_path) # "tu_modelo.onnx"
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # 2. Guardar el directorio de trabajo actual
    original_cwd = os.getcwd()

    try:
        # 3. Cambiar el directorio de trabajo actual al destino para que el exportador de C++
        # no vea ninguna ruta con barras inclinadas ni caracteres conflictivos en Windows.
        os.chdir(dest_dir)
        
        # Eliminar archivos existentes si los hay
        if os.path.exists(filename):
            try: os.remove(filename)
            except Exception: pass
        data_filename = filename + ".data"
        if os.path.exists(data_filename):
            try: os.remove(data_filename)
            except Exception: pass

        # 4. Exportar usando el nombre de archivo plano nativo
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=18,  # Especificado por la pizarra como versión v12+ (18 >= 12)
            do_constant_folding=True,
            input_names=["cam_input"],
            output_names=["confidence_score"],
            dynamic_axes={
                "cam_input": {0: "batch_size"},
                "confidence_score": {0: "batch_size"}
            }
        )
        print("[+] Grafo ONNX generado exitosamente de forma nativa en el destino.")
        
        # 5. Regresar al directorio de ejecución original de forma segura
        os.chdir(original_cwd)
        print(f"[+] Modelo ONNX guardado exitosamente en: '{final_onnx_path}'")
        
        # Validar el modelo usando la API de ONNX
        import onnx
        onnx_model = onnx.load(final_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[+] Validación de ONNX completada: Grafo sintáctica y estructuralmente válido.")
        return True
    except Exception as e:
        print(f"[ERROR] Fallo al exportar/validar ONNX: {e}")
        # Asegurar regresar al directorio original en caso de fallo
        try: os.chdir(original_cwd)
        except Exception: pass
        return False


def main():
    try:
        X, y = load_dataset()
        model = train_cnn(X, y, epochs=20, lr=0.001, batch_size=32)
        export_to_onnx(model)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
