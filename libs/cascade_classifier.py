import cv2

class FaceDetector:
    def __init__(self, xml_path):
        self.face_cascade = cv2.CascadeClassifier(xml_path)
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces[0] # x, y, w, h
