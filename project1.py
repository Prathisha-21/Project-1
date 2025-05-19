import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


image = cv2.imread('crowd image.jpg')  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


face_count = len(faces)
cv2.putText(image, f'Crowd Count: {face_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Crowd Count from Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()