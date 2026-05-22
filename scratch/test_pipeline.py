import cv2
import os
import numpy as np
import onnxruntime as ort

model_path = "./model/tu_modelo.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)

def evaluate_crop(img, x, y, w, h):
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
    
    # Grid search around candidate center and size
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
                
                score, box = evaluate_crop(img, new_x, new_y, size, size)
                if score > best_score:
                    best_score = score
                    best_box = box
                    
    return best_score, best_box

def run_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return
    h_img, w_img, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Relaxed face detector
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
    print(f"\n--- {img_path} ---")
    print(f"Faces detected: {len(faces)}")
    
    all_eye_candidates = []
    
    if len(faces) > 0:
        for i, (fx, fy, fw, fh) in enumerate(faces):
            # Upper 60% of face for eyes
            roi_gray = gray[fy:fy + int(fh*0.65), fx:fx + fw]
            
            # Sensitive eye detector inside face ROI
            eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
                
            eyes_glass = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes_glass:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
    else:
        # Fallback to global search if no face found (with standard parameters to avoid noise)
        eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes_std:
            all_eye_candidates.append((ex, ey, ew, eh))
            
        eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes_glass:
            all_eye_candidates.append((ex, ey, ew, eh))
            
    # Deduplicate candidate boxes
    unique = []
    for x, y, w, h in all_eye_candidates:
        dup = False
        for ux, uy, uw, uh in unique:
            dist = np.sqrt(((x+w/2) - (ux+uw/2))**2 + ((y+h/2) - (uy+uh/2))**2)
            if dist < min(w, uw) * 0.5:
                dup = True
                break
        if not dup:
            unique.append((x, y, w, h))
            
    print(f"Candidates before refinement: {len(unique)}")
    
    # Evaluate and refine each candidate
    refined_results = []
    for x, y, w, h in unique:
        score, refined_box = refine_box(img, x, y, w, h)
        refined_results.append((score, refined_box))
        
    # Sort by score descending
    refined_results.sort(key=lambda x: x[0], reverse=True)
    
    for i, (score, box) in enumerate(refined_results):
        print(f"  Cand {i+1}: Score={score:.6f} at {list(box)}")

for i in range(1, 7):
    run_pipeline(f"ojos2/{i}.jpg")
