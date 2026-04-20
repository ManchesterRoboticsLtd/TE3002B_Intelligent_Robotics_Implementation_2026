import cv2 as cv
import numpy as np
 

#HSV (Hue, Saturation, Value) in OpenCV
#The values are in the ranges of H /in [0,180] deg, S /in [0,255] and V /in [0,255]
    
#HSV Values for Red Colour 
#Red is on the 180 degrees limit in thr HSV so two ranges must be done 
#from 160 to 0 and from 0 to 20
H_red_low_limit = 179
S_red_low_limit = 95
V_red_low_limit = 130

H_red_up_limit = 5
S_red_up_limit = 95
V_red_up_limit = 130

#HSV Values for Blue Colour 
H_blue_low_limit = 95
S_blue_low_limit = 90
V_blue_low_limit = 90

H_blue_up_limit = 110
S_blue_up_limit = 255
V_blue_up_limit = 255


if __name__=="__main__":
    #cap = cv.VideoCapture(2)
    video_path = 'videos/sample_video.mp4'
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open camera")

    #HSV Values for Green Colour
    while True:

        # Reading frame from video or webcam
        # and resizing to match when combining them
        ret, image = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        k = cv.waitKey(25) & 0xFF
        if k == 27:
            break

        # Convert BGR to HSV
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        #Red Ranges
        #Red colour requires 2 ranges because its on the limits of the angles [min_H, 179] to [0, max_H]
        red_lower_limit1 = np.array([H_red_low_limit, S_red_low_limit, V_red_low_limit])
        red_upper_limit1 = np.array([179, 255, 255])

        red_lower_limit2 = np.array([0, S_red_up_limit, V_red_up_limit])
        red_upper_limit2 = np.array([H_red_up_limit, 255, 255])

        #Blue Ranges
        blue_lower_limit = np.array([H_blue_low_limit, S_blue_low_limit, V_blue_low_limit])
        blue_upper_limit = np.array([H_blue_up_limit, S_blue_up_limit, V_blue_up_limit])

        #Red Masks
        red_mask1 = cv.inRange(hsv_image, red_lower_limit1, red_upper_limit1)
        red_mask2 = cv.inRange(hsv_image, red_lower_limit2, red_upper_limit2)

        #Blue Mask
        blue_mask = cv.inRange(hsv_image, blue_lower_limit, blue_upper_limit)

        #Partial masks applied to the image (AND Function with the image)
        masked_red1 = cv.bitwise_and(image,image,mask=red_mask1)
        masked_red2 = cv.bitwise_and(image,image,mask=red_mask2)
        
        #Full red mask applied to te image
        red_img_result = cv.bitwise_or(masked_red1, masked_red2)

        #Full blue mask applied to te image
        blue_img_result = cv.bitwise_and(image,image,mask=blue_mask)

        #Transform to Grayscale
        bw_blue_res = cv.cvtColor(blue_img_result, cv.COLOR_BGR2GRAY)
        bw_red_res = cv.cvtColor(red_img_result, cv.COLOR_BGR2GRAY)

        _ , red_res = cv.threshold(bw_red_res, 30, 255, cv.THRESH_BINARY_INV) 
        _ , blue_res = cv.threshold(bw_blue_res, 33, 255, cv.THRESH_BINARY_INV) 

        kernel = np.ones((5, 5), np.uint8)
        red_res = cv.dilate(red_res, kernel, iterations=3)
        red_res = cv.erode(red_res, kernel, iterations=3)

        blue_res = cv.dilate(blue_res, kernel, iterations=8)
        blue_res = cv.erode(blue_res, kernel, iterations=8)

        cv.imshow('image', image)
        cv.imshow('red mask', red_img_result)
        cv.imshow('blue mask', blue_img_result)
        cv.imshow('blue res', blue_res)
        cv.imshow('red res', red_res)

    # When everything done, release the captureq
    cap.release()
    cv.destroyAllWindows()

