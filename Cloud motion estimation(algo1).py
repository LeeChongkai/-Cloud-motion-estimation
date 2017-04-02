import cv2
import numpy as np

e1 = cv2.getTickCount()
frame1 = cv2.imread('2016-09-03-10-08-25-wahrsis3.1000x667_predicted_algo1.jpg',1)
frame2 = cv2.imread('2016-09-03-10-10-25-wahrsis3.1000x667_predicted_algo1.jpg',1)


prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

frame3 = np.zeros_like(frame1) #normalize an empty array

flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

flowY = flow[...,0]
flowX = flow[...,1]

n = np.shape(frame1)[0]
m = np.shape(frame1)[1]

#predict third image
for x in range(0,n):
    for y in range(0,m):
        value = frame2[x,y]
        shiftX = flowX[x,y]
        shiftY = flowY[x,y]
        frame3[x + int(shiftX) , y + int(shiftY)] = value

#Using neighbour pixel value to fill up the empty pixel
for a in range (0,n):
    for b in range(0,m):
        k = frame3[a,b]
        
        if np.all(k == 0):
            if ((a+1) > (n-1)):
                a1 = frame3[a,b]
            else:
                a1 = frame3[(a+1),b]
                
            if ((a-1) < 0):
                a2 = frame3[a,b]
            else:
                a2 = frame3[(a-1),b]

            if ((b+1) > (n-1)):
                b1 = frame3[a,b]
            else:
                
                b1 = frame3[a,(b+1)]
                
            if ((b-1) < 0 ):
                b2 = frame3[a,b]
            else:
                b2 = frame3[a,(b-1)]

            a1 = a1.astype('uint16')
            a2 = a2.astype('uint16')
            b1 = b1.astype('uint16')
            b2 = b2.astype('uint16')
            
            avg_value = (a1 + a2 + b1 + b2)/np.max([np.count_nonzero([np.sum(a1), np.sum(a2), np.sum(b1), np.sum(b2)]), 1])
            
            frame3[a,b] = avg_value.astype('uint8')
            
   
cv2.imwrite('2016-09-03-10-12-25-wahrsis3.1000x667_predicted_algo1.jpg', frame3)
cv2.imshow('Opticalflow',frame3)


e2 = cv2.getTickCount()
time = (e2-e1) / cv2.getTickFrequency()
print time

k = cv2.waitKey(0) & 0xff
if k == ord('s'):
    cv2.imwrite('cloud13.jpg', frame3)
    cv2.imshow('Opticalflow',frame3)

cv2.destroyAllWindows()



