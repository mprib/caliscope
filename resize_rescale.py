import cv2 as cv

photo = "Photos\IMG_20140526_224627656.jpg"
video =   "anipose_videos\calibration\calib-charuco-camA.MOV"

def rescaleFrame(frame, scale=0.75):
    """ 
    takes a frame and returns a scaled frame
    will work for photos, videos, and live video
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)




def ResizeVideo():


    capture = cv.VideoCapture(video)

    while True:
        isTrue, frame = capture.read()

        frame_resized = rescaleFrame(frame, scale=0.2)

        cv.imshow("Original Video", frame)
        cv.imshow("Rescaled Video", frame_resized)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break


def ResizeImage():

    img = cv.imread(photo)
    
    resized_image = rescaleFrame(img)

    img_resized = rescaleFrame(img, scale=0.2)

    cv.imshow("Original", img)
    cv.imshow("Rescaled ", img_resized)
 
    cv.waitKey(0)

#### ALTERNATE WAY TO GO ABOUT RESIZING


def ResizeLiveFeed():
    """ 
    takes in two video feeds, though one is displayed with a different size
    note that few options are available for size: 

    '320.0x240.0': 'OK', 
    '640.0x480.0': 'OK', 
    '1280.0x720.0': 'OK'


    Attempting to go lower than the default value created problems

    Also note that just resizing each frame is more flexible. Just don't even 
    bother using this
    """
    capture = cv.VideoCapture(0)
    capture_resized = cv.VideoCapture(1)

    width = 1280
    height = 720

    capture_resized.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture_resized.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        isTrue, frame = capture.read()


        isTrue, frame_resized = capture_resized.read()

        cv.imshow("Original Video", frame)
        cv.imshow("Rescaled Video", frame_resized)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break



if __name__ == "__main__":
    # ResizeImage()
    ResizeLiveFeed()