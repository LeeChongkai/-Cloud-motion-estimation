import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import exifread
import bisect

def CalSolarRad(image,image1):

    b,g,r = cv2.split(image1)

    #Locate the sun
    n = np.shape(r)[0]
    m = np.shape(r)[1]
    p = []
    for x in range(0,n):
        for y in range(0,m):
            j = r[x,y]

            if np.all(j > 240):
                p.append([x,y])
                
    
    arr = np.array([p])
    avg = arr.mean(axis=1)
    
    arr1 = avg[0]
    print arr1
    
    
    around_sun = r[(int(arr1[0] - 75)):(int(arr1[0] + 75)),(int(arr1[1] - 75)):(int(arr1[1] + 75))]
    cv2.imshow('around_sun',around_sun)

    
    k = cv2.waitKey(0) & 0xff
    if k == ord('s'):
        cv2.imwrite('cloud13.jpg', frame3)
        cv2.imshow('Opticalflow',frame3)

    cv2.destroyAllWindows()
    
    plt.figure(1)
    plt.imshow(around_sun)
    plt.show()

    
    N = np.mean(around_sun)   

    exp_time1 = float(0.00125)
    ISO1 = 800
    aperture1 = 13
    
#Compute Luminance ---------------------------
    lum = float(N*(aperture1**2 / (ISO1*exp_time1)))
    print lum

    print 'The relative luminance is :',(str(lum))
    
#Estimate Solar Radiation---------------------------
    solarRad = 0.0138*lum - 39.896

    print 'Solar Rad is :',(str(solarRad)), 'W/m2'

    return(solarRad)


    

    

    

    


