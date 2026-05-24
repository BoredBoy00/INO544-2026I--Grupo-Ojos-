import os
import glob
import cv2
import numpy as np
import onnx
from onnx import helper, TensorProto

TRAIN_SIZE = (24, 24)
INFER_SIZE = 224
RNG = np.random.default_rng(42)


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


def crop_bgr_to_rgb(img_bgr, x, y, w, h):
    x, y, w, h = clamp_box(x, y, w, h, img_bgr.shape[1], img_bgr.shape[0])
    crop = img_bgr[y : y + h, x : x + w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def rgb_to_gray(img_rgb):
    return 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]


def patch_to_features(crop_rgb):
    low_rgb = cv2.resize(crop_rgb, TRAIN_SIZE).astype(np.float32) / 255.0
    low_gray = rgb_to_gray(low_rgb)
    return low_gray.flatten()


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
            pos_crops.append(crop_bgr_to_rgb(img, ex, ey, ew, eh))
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

    X_pos = np.array([patch_to_features(c) for c in pos_crops], dtype=np.float32)
    X_neg = np.array([patch_to_features(c) for c in neg_crops], dtype=np.float32)

    X = np.vstack((X_pos, X_neg))
    y = np.vstack(
        (
            np.ones((pos_count, 1), dtype=np.float32),
            np.zeros((neg_count, 1), dtype=np.float32),
        )
    )

    indices = np.arange(X.shape[0])
    RNG.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Mezclar listas de crops para calibración (misma semilla)
    all_crops = pos_crops + neg_crops
    crop_labels = [1] * pos_count + [0] * neg_count
    perm = np.arange(len(all_crops))
    RNG.shuffle(perm)
    cal_crops = [all_crops[i] for i in perm]
    cal_labels = [crop_labels[i] for i in perm]

    return X, y, cal_crops, cal_labels


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_classifier(X, y, epochs=250, lr=0.08, lambda_reg=0.04):
    m, n = X.shape
    print(f"[*] Entrenando clasificador (muestras: {m}, dims: {n})...")

    W = np.zeros((n, 1), dtype=np.float32)
    b = np.zeros((1, 1), dtype=np.float32)

    for epoch in range(1, epochs + 1):
        z = np.dot(X, W) + b
        a = sigmoid(z)
        loss = (
            -np.mean(y * np.log(a + 1e-15) + (1.0 - y) * np.log(1.0 - a + 1e-15))
            + (lambda_reg / (2 * m)) * np.sum(W ** 2)
        )
        dz = a - y
        dW = (1.0 / m) * np.dot(X.T, dz) + (lambda_reg / m) * W
        db = (1.0 / m) * np.sum(dz, axis=0, keepdims=True)
        W -= lr * dW
        b -= lr * db

        predictions = (a > 0.5).astype(np.float32)
        accuracy = np.mean(predictions == y) * 100.0

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"    Época {epoch:03d}/{epochs:03d} | "
                f"Pérdida: {loss:.6f} | Precisión: {accuracy:.2f}%"
            )

    print("[+] Entrenamiento completado.")
    return W, b


def build_weight_vector(W_low):
    """Pesos ONNX sin calibración A/B (solo escala 24→224)."""
    W_img = W_low.reshape(TRAIN_SIZE)
    W_up = cv2.resize(W_img, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
    W_3d = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.float32)
    W_3d[:, :, 0] = 0.299 * W_up
    W_3d[:, :, 1] = 0.587 * W_up
    W_3d[:, :, 2] = 0.114 * W_up
    D = INFER_SIZE / float(TRAIN_SIZE[0])
    scale_factor = 1.0 / (D ** 2)
    return (W_3d.flatten() * scale_factor).astype(np.float32)


def score_crop_224(crop_rgb, W_vec, b_scalar):
    """Misma lógica que app.py / ONNX en inferencia."""
    resized = cv2.resize(crop_rgb, (INFER_SIZE, INFER_SIZE)).astype(np.float32) / 255.0
    flat = resized.flatten()
    logit = float(np.dot(flat, W_vec) + b_scalar)
    return sigmoid(logit), logit


def compute_calibration(W_low, b_raw, cal_crops, cal_labels):
    """
    Calibra A y B en logit' = A * logit + B usando parches a 224×224
    (idéntico al pipeline de la cámara).
    """
    W_base = build_weight_vector(W_low)
    b_base = float(b_raw[0][0])

    pos_logits = []
    neg_logits = []
    for crop, label in zip(cal_crops, cal_labels):
        _, logit = score_crop_224(crop, W_base, b_base)
        if label == 1:
            pos_logits.append(logit)
        else:
            neg_logits.append(logit)

    if not pos_logits or not neg_logits:
        print("[!] Calibración: datos insuficientes, usando valores por defecto.")
        return 2.5, 0.0

    mu_pos = float(np.median(pos_logits))
    mu_neg = float(np.median(neg_logits))
    p95_neg = float(np.percentile(neg_logits, 95))
    target_pos = 6.0
    target_neg = -6.0

    spread = mu_pos - mu_neg
    if abs(spread) < 0.05:
        print("[!] Positivos y negativos muy solapados en logit; revisar dataset.")
        A, B = 2.0, 0.0
    else:
        A = (target_pos - target_neg) / spread
        B_median = target_pos - A * mu_pos
        target_p95_score = 0.12
        target_p95_logit = float(np.log(target_p95_score / (1.0 - target_p95_score)))
        B_strict = target_p95_logit - A * p95_neg

        def metrics_for_b(b_val):
            pos_s = [sigmoid(A * lp + b_val) for lp in pos_logits]
            neg_s = [sigmoid(A * ln + b_val) for ln in neg_logits]
            return float(np.median(pos_s)), float(np.percentile(neg_s, 95))

        med_m, p95_m = metrics_for_b(B_median)
        med_s, p95_s = metrics_for_b(B_strict)

        # Priorizar ojos >0.92; si el ajuste estricto lo rompe, usar mezcla
        if med_s >= 0.92 and p95_s <= 0.15:
            B = B_strict
        elif med_m >= 0.92 and p95_m <= 0.20:
            B = B_median
        else:
            B = 0.6 * B_median + 0.4 * B_strict

    def cal_score(crop):
        _, raw = score_crop_224(crop, W_base, b_base)
        return sigmoid(A * raw + B)

    pos_scores = [cal_score(c) for c, l in zip(cal_crops, cal_labels) if l == 1]
    neg_scores = [cal_score(c) for c, l in zip(cal_crops, cal_labels) if l == 0]

    print("[*] --- Calibración 224×224 (validación interna) ---")
    print(f"    Logit mediano ojo:     {mu_pos:.4f}")
    print(f"    Logit mediano no-ojo:  {mu_neg:.4f}")
    print(f"    Logit p95 no-ojo:      {p95_neg:.4f}")
    print(f"    A={A:.4f}, B={B:.4f}")
    print(f"    Score mediano ojo:     {np.median(pos_scores):.4f}")
    print(f"    Score mediano no-ojo:  {np.median(neg_scores):.4f}")
    print(f"    Score p95 no-ojo:      {np.percentile(neg_scores, 95):.4f}")

    return A, B


def export_to_onnx(W_low, b, cal_crops, cal_labels, output_path="./model/tu_modelo.onnx"):
    A, B = compute_calibration(W_low, b, cal_crops, cal_labels)

    W_base = build_weight_vector(W_low)
    W_final = (W_base * A).reshape(-1, 1)
    b_calibrated = float(b[0][0]) * A + B

    print(f"[*] Exportando ONNX a '{output_path}'")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cam_input = helper.make_tensor_value_info(
        "cam_input", TensorProto.FLOAT, [1, INFER_SIZE, INFER_SIZE, 3]
    )
    confidence_score = helper.make_tensor_value_info(
        "confidence_score", TensorProto.FLOAT, [1, 1]
    )

    w_initializer = helper.make_tensor(
        name="model_weights",
        data_type=TensorProto.FLOAT,
        dims=[INFER_SIZE * INFER_SIZE * 3, 1],
        vals=W_final.flatten().tolist(),
    )
    b_initializer = helper.make_tensor(
        name="model_bias",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[b_calibrated],
    )

    flatten_node = helper.make_node(
        "Flatten", inputs=["cam_input"], outputs=["flat_input"], axis=1
    )
    matmul_node = helper.make_node(
        "MatMul", inputs=["flat_input", "model_weights"], outputs=["matmul_out"]
    )
    add_node = helper.make_node(
        "Add", inputs=["matmul_out", "model_bias"], outputs=["logits"]
    )
    sigmoid_node = helper.make_node(
        "Sigmoid", inputs=["logits"], outputs=["confidence_score"]
    )

    graph_def = helper.make_graph(
        nodes=[flatten_node, matmul_node, add_node, sigmoid_node],
        name="EyeClassifierNetwork",
        inputs=[cam_input],
        outputs=[confidence_score],
        initializer=[w_initializer, b_initializer],
    )

    opset = helper.make_operatorsetid("", 12)
    model_def = helper.make_model(
        graph_def, producer_name="numpy-eye-trainer-v2", opset_imports=[opset]
    )

    try:
        onnx.checker.check_model(model_def)
        print("[+] Grafo ONNX válido.")
    except Exception as e:
        print(f"[ERROR] Validación ONNX: {e}")
        return False

    onnx.save(model_def, output_path)
    print(f"[+] Modelo guardado en '{output_path}'")
    return True


def main():
    try:
        X, y, cal_crops, cal_labels = load_dataset()
        W, b = train_classifier(X, y)
        export_to_onnx(W, b, cal_crops, cal_labels)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
