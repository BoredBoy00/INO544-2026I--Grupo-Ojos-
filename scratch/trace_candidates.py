import cv2
import os
import numpy as np

face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)

def trace_image(img_path):
    img_bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    print(f"\n=== TRACE FOR {img_path} ===")
    print(f"Faces: {list(faces)}")
    
    all_eye_candidates = []
    
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy + int(fh*0.6), fx:fx + fw]
            
            eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            print(f"  Face {fx},{fy},{fw},{fh} - eyes_std: {list(eyes_std)}")
            for (ex, ey, ew, eh) in eyes_std:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
                
            eyes_glass = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            print(f"  Face {fx},{fy},{fw},{fh} - eyes_glass: {list(eyes_glass)}")
            for (ex, ey, ew, eh) in eyes_glass:
                all_eye_candidates.append((fx + ex, fy + ey, ew, eh))
    else:
        eyes_std = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        print(f"  Global eyes_std: {list(eyes_std)}")
        for (ex, ey, ew, eh) in eyes_std:
            all_eye_candidates.append((ex, ey, ew, eh))
            
        eyes_glass = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        print(f"  Global eyes_glass: {list(eyes_glass)}")
        for (ex, ey, ew, eh) in eyes_glass:
            all_eye_candidates.append((ex, ey, ew, eh))
            
    print(f"Before NMS: {all_eye_candidates}")
    
    # NMS
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
    print(f"After NMS: {unique_candidates}")

trace_image("ojos2/2.jpg")
trace_image("ojos2/4.jpg")
