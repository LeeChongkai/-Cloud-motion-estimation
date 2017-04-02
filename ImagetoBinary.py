# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def compare_Binary_images(imageA, imageB,im_mask):

#Convert image to binary image ----------------------------------------------
    im_mask[im_mask<128]=0
    im_mask[im_mask>128]=255

    n = np.shape(imageA)[0]
    m = np.shape(imageA)[1]
    arr1 = []
    arr2 = []
    for x in range(0,n):
        for y in range(0,m):
            j = im_mask[x,y]           
            if np.all(j == 255):
                arr1.append(r1[x,y])
                arr2.append(r2[x,y])
            
    arr1 = np.array([arr1])
    arr2 = np.array([arr2])
    #To obtain the threhold -----------------------------------------------
    thresh1, bw_im1 = cv2.threshold(arr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh2, bw_im2 = cv2.threshold(arr2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    for c in range(0,n):
        for k in range(0,m):
            j = im_mask[c,k]
            if np.all(j == 255):
                if np.all(r1[c,k] < thresh1):
                    r1[c,k] = 0
                else:
                    r1[c,k] = 255
                    
                if np.all(r2[c,k] < thresh2):
                    r2[c,k] = 0
                else:
                    r2[c,k] = 255
   
    
#Calculate Error btw original & predicted image -----------------------------
    no_of_pixel = 0
    no_of_diff_pixel = 0
    for a in range(0,n):
        for b in range(0,m):
            k = im_mask[a,b]           
            if np.all(k == 255):
                no_of_pixel += 1
                if np.all(r1[a,b] != r2[a,b]):
                    no_of_diff_pixel += 1

    Error = (float(no_of_diff_pixel) / float(no_of_pixel))*100
    print 'Error =', Error,'%'
    
    
    # setup the figure
    fig = plt.figure()
    plt.suptitle("Error: %.2f" % (Error))
 
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(r1, cmap = plt.cm.gray)
    plt.axis("off")
 
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(r2, cmap = plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()

    cv2.imwrite("cloud12_binary.jpg",r1)
    cv2.imwrite("cloud12_binary.jpg",r2)
    

# Load image ---------------------------------------------------------------
original = cv2.imread('cloud12.jpg',1)
predicted = cv2.imread('cloud12.jpg',1)
b1,g1,r1 = cv2.split(original)
b2,g2,r2 = cv2.split(predicted)

im_mask = cv2.imread("mask.png",1)
im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

# Call function to compare images-------------------------------------------
compare_Binary_images(r1, r2, im_mask)


	
print "end"


k = cv2.waitKey(0) & 0xff
if k == ord('s'):
    cv2.imwrite('cloud13.jpg', frame3)
    cv2.imshow('Opticalflow',frame3)

cv2.destroyAllWindows()

