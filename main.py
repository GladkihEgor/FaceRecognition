from face_recognition import FaceRecognition
from methods import FaceRecognitionMethod
from view import View

if __name__ == '__main__':
    view = View()
    face_recognition = FaceRecognition(view)
    face_recognition.run(FaceRecognitionMethod.DETECTOR)  # Remove method if want use haar cascade
