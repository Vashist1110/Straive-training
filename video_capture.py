import cv2
def func(a):
    cap=cv2.VideoCapture(a)
    while True:
        check,output=cap.read()
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces=faceCascade.detectMultiScale(output,1.3,5)
        for (x,y,w,h) in faces:
            output = cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)
            output = cv2.resize(output, (360,240))
            cv2.imshow('face recogonition',output)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
func('sample_video.avi')