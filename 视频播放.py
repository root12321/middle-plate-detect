import cv2
import numpy as np
import pandas as pd
def Cvpointgray(img):
    x=[]
    x0=0;y0=0
    for i in range(img.shape[1]):
        for j in range(1,img.shape[0]):
            x0=x0+img[j,i]
            y0=y0+img[j,i]*j
        y=y0/x0
        if x0==0:
            y=0
        y=round(y)
        x.append(y)
    return x
cap = cv2.VideoCapture("D:\opencv_camera\digits\shiyan\output2.avi")
a=[]
while(cap.isOpened()):
    # get a frame
    ret, frame = cap.read()
    frame = frame[100:400,300:1000]
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    emptyImage3 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(emptyImage3, 80, 255, cv2.THRESH_BINARY)
    # show a frame
    cv2.imshow("capture",th1)

    a.append(Cvpointgray(th1))
    #print(Cvpointgray(th1))
    if cv2.waitKey(100) & 0xFF == ord('q'):#视频播放完以后 直接按下‘q’,打断循环。
        break
#print(a)
cap.release()
cv2.destroyAllWindows()

result=np.array(a)

data1 = pd.DataFrame(result)
#data1.to_csv('data1.csv')