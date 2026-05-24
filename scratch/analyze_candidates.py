import cv2
import os

face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
eye_glass_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
eye_glass_cascade = cv2.CascadeClassifier(eye_glass_path)

def analyze(img_name):
    img_path = f"ojos2/{img_name}"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n--- Analyzing {img_name} ({w}x{h}) ---")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    print(f"Detected Faces: {len(faces)}")
    for i, (fx, fy, fw, fh) in enumerate(faces):
        print(f"  Face {i}: x={fx}, y={fy}, w={fw}, h={fh}")
        roi_gray = gray[fy:fy + int(fh*0.6), fx:fx + fw]
        eyes_std = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        print(f"    Eyes standard in face ROI: {len(eyes_std)}")
        for (ex, ey, ew, eh) in eyes_std:
            print(f"      Eye std: x={fx+ex}, y={fy+ey}, w={ew}, h={eh}")
        eyes_glass = eye_glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        print(f"    Eyes glass in face ROI: {len(eyes_glass)}")
        for (ex, ey, ew, eh) in eyes_glass:
            print(f"      Eye glass: x={fx+ex}, y={fy+ey}, w={ew}, h={eh}")

    # Detect eyes on the entire image
    eyes_std_global = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    print(f"Global Eyes standard: {len(eyes_std_global)}")
    for (ex, ey, ew, eh) in eyes_std_global:
        print(f"  Eye std: x={ex}, y={ey}, w={ew}, h={eh}")
    eyes_glass_global = eye_glass_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    print(f"Global Eyes glass: {len(eyes_glass_global)}")
    for (ex, ey, ew, eh) in eyes_glass_global:
        print(f"  Eye glass: x={ex}, y={ey}, w={ew}, h={eh}")

analyze("2.jpg")
analyze("4.jpg")
