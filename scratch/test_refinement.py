import cv2
import os
import numpy as np
import onnxruntime as ort

model_path = "./model/tu_modelo.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def evaluate_crop(img, x, y, w, h):
    # Ensure coordinates are within image boundaries
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

def refine_box(img, x, y, w, h):
    best_score = -1.0
    best_box = (x, y, w, h)
    
    # We will search in a local grid:
    # dx: from -12 to 12 with step of 2
    # dy: from -12 to 12 with step of 2
    # ds: size variation from -8 to 8 with step of 2 (maintaining square aspect ratio)
    for ds in range(-8, 9, 2):
        size = w + ds
        if size < 20:
            continue
        # We want to center the new box relative to the old one's center, plus an offset dx, dy
        old_cx = x + w/2
        old_cy = y + h/2
        
        for dy in range(-12, 13, 2):
            for dx in range(-12, 13, 2):
                new_x = int(old_cx - size/2 + dx)
                new_y = int(old_cy - size/2 + dy)
                
                score, box = evaluate_crop(img, new_x, new_y, size, size)
                if score > best_score:
                    best_score = score
                    best_box = box
                    
    return best_score, best_box

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    print(f"\n===== Refinement for {img_path} =====")
    
    candidates = []
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy + int(fh*0.6), fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                candidates.append((fx + ex, fy + ey, ew, eh))
    else:
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            candidates.append((ex, ey, ew, eh))
            
    # Deduplicate candidates
    unique = []
    for x, y, w, h in candidates:
        dup = False
        for ux, uy, uw, uh in unique:
            dist = np.sqrt(((x+w/2) - (ux+uw/2))**2 + ((y+h/2) - (uy+uh/2))**2)
            if dist < min(w, uw) * 0.5:
                dup = True
                break
        if not dup:
            unique.append((x, y, w, h))
            
    for i, (x, y, w, h) in enumerate(unique):
        orig_score, _ = evaluate_crop(img, x, y, w, h)
        ref_score, ref_box = refine_box(img, x, y, w, h)
        print(f"Candidate {i+1}:")
        print(f"  Original:  [{x}, {y}, {w}, {h}] -> Score: {orig_score:.6f}")
        print(f"  Refined:   {list(ref_box)} -> Score: {ref_score:.6f}")

process_image("ojos2/2.jpg")
process_image("ojos2/4.jpg")
