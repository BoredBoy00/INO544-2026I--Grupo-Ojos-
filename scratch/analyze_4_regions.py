import cv2
import os

img = cv2.imread("ojos2/4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

print("Faces detected in 4.jpg with relaxed parameters:")
for i, (fx, fy, fw, fh) in enumerate(faces):
    print(f"  Face {i}: [{fx}, {fy}, {fw}, {fh}]")
    
# Let's save a context image for Candidate 1: [302, 548, 22, 22]
# We crop a 100x100 region centered at Candidate 1
c1_ctx = img[max(0, 548-40):min(img.shape[0], 548+62), max(0, 302-40):min(img.shape[1], 302+62)]
cv2.imwrite("C:/Users/JhonOcz/.gemini/antigravity-ide/brain/c571f64c-0dc4-4ab1-9ad3-67b150bd97af/scratch/crop_4_cand1_context.jpg", c1_ctx)

# Let's save a context image for Candidate 2: [358, 271, 44, 44]
# We crop a 150x150 region centered at Candidate 2
c2_ctx = img[max(0, 271-50):min(img.shape[0], 271+94), max(0, 358-50):min(img.shape[1], 358+94)]
cv2.imwrite("C:/Users/JhonOcz/.gemini/antigravity-ide/brain/c571f64c-0dc4-4ab1-9ad3-67b150bd97af/scratch/crop_4_cand2_context.jpg", c2_ctx)

print("Saved context crops.")
