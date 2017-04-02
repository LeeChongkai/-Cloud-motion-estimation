# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def RMSE_mask(imageA,imageB,im_mask):
 
    im_mask[im_mask<128]=0
    im_mask[im_mask>128]=255
 
 
    mask_st = im_mask.flatten()
    imA_st = imageA.flatten()
    imB_st = imageB.flatten()
 
    whichIndex = np.where(mask_st==255)
    arr1 = imA_st[whichIndex]
    arr2 = imB_st[whichIndex]
    n = arr1.size
 
    RMSE = np.sqrt((((arr1-arr2)**2).sum())/n)
    print ('RMSE=',RMSE)

    return(RMSE)
 	
 
def compare_images(imageA, imageB, title):
	# compute the root mean squared error for the images 
	r = RMSE_mask(imageA,imageB,im_mask)
	
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("RMSE: %.2f" % (r))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()
	
# load the images -- the original, predicted and mask
original = cv2.imread("cloud17.jpg",1)
predicted = cv2.imread("cloud17_predicted_algo2.jpg",1)
im_mask = cv2.imread("mask.png",1)


# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)


# compare the images
compare_images(original, predicted, "Original vs. Predicted")


	
print "end"


k = cv2.waitKey(0) & 0xff
if k == ord('s'):
    cv2.imwrite('cloud13.jpg', frame3)
    cv2.imshow('Opticalflow',frame3)

cv2.destroyAllWindows()
