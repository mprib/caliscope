import cv2 as cv
from resize_rescale import rescaleFrame


photo = "Photos\IMG_20140526_224627656.jpg"
video =   "anipose_videos\calibration\calib-charuco-camA.MOV"

img = cv.imread(photo)
print(img.shape)

img = rescaleFrame(img, 0.3)

cv.imshow("ScaledHoneymoon", img)
print(img.shape)

# Convert to gray
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Honeymoon", gray_img)


# blur
blur = cv.GaussianBlur(img,(3,3), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)

# edge cascade
# Canny is apparently an old and well known algorithm for edge detection.
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edge Detection", canny)

# This edge cascaded image becomes the "structuring element" in the processing below, 
# whatever that means.
canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny Blurred Edge Detection", canny)


# Dilate the image
# Dilation appears to make the edges thicker and simpler
dilated = cv.dilate(canny, (3,3),iterations=3)
cv.imshow("Dilated", dilated)


# Erode the image
# eroding an image may be able to take a dilated image and return it to an uneroded state (?)
# What is all this kernal size stuff? (the tuple in the second argument)
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow("Eroded", eroded)

# eroding kinda restored it, but did better with more iterations.



# resizing without respect to aspect ratio
resized = cv.resize(img, (500,500))     
cv.imshow("Resized", resized)


# resizing and interpolating
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)     
cv.imshow("Interpolated Resized", resized)

# Cropping
# images are just arrays, so slice into them to crop
# note: height is before width (alphabetical).
cropped = img[0:350, 100:350]
cv.imshow("Cropped", cropped)


# where I'm leaving off now:
# https://youtu.be/oXlwWbU8l2o?t=2654

cv.waitKey(0)