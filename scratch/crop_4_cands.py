import cv2

img = cv2.imread("ojos2/4.jpg")
if img is not None:
    # Candidate 1: [302, 548, 22, 22]
    c1 = img[548:548+22, 302:302+22]
    cv2.imwrite("C:/Users/JhonOcz/.gemini/antigravity-ide/brain/c571f64c-0dc4-4ab1-9ad3-67b150bd97af/scratch/crop_4_cand1.jpg", c1)
    
    # Candidate 2: [358, 271, 44, 44]
    c2 = img[271:271+44, 358:358+44]
    cv2.imwrite("C:/Users/JhonOcz/.gemini/antigravity-ide/brain/c571f64c-0dc4-4ab1-9ad3-67b150bd97af/scratch/crop_4_cand2.jpg", c2)
    print("Cropped and saved candidates for 4.jpg")
else:
    print("Could not load 4.jpg")
