
from CalSolarRad_NEW import *

# Input for EXIF parameter
image = './cloud22.jpg'

# To locate Sun
image1 = cv2.imread('cloud22.jpg',1)
solarRad = CalSolarRad(image,image1)




