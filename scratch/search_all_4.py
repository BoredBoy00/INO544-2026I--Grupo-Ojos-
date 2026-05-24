import cv2
import os
import numpy as np
import onnxruntime as ort

model_path = "./model/tu_modelo.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

img = cv2.imread("ojos2/4.jpg")
h_img, w_img, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
eye_cascade = cv2.CascadeClassifier(eye_path)
eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)

print("--- Testing scaleFactor and minNeighbors on 4.jpg ---")
for sf in [1.05, 1.1, 1.15, 1.2]:
    for mn in [2, 3, 4, 5]:
        eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(20, 20))
        eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(20, 20))
        
        candidates = []
        for x, y, w, h in list(eyes_std) + list(eyes_glass):
            candidates.append((x, y, w, h))
            
        # Deduplicate
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
                
        # Evaluate top unique candidates
        evaluated = []
        for x, y, w, h in unique:
            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            it = crop_resized.astype(np.float32) / 255.0
            it = np.expand_dims(it, axis=0)
            score = float(session.run([output_name], {input_name: it})[0][0][0])
            evaluated.append((score, x, y, w, h))
            
        high_score_cands = [e for e in evaluated if e[0] >= 0.5]
        if high_score_cands:
            print(f"sf={sf}, mn={mn}: Found {len(high_score_cands)} candidates with score >= 0.5:")
            for e in high_score_cands:
                print(f"  Score: {e[0]:.6f} at [{e[1]}, {e[2]}, {e[3]}, {e[4]}]")
