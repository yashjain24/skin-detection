from pyimagesearch import imutils
import numpy as np
import argparse
import cv2

#construct arguement
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
	help = "path to the video file (optional)")
args = vars(ap.parse_args())

#define upper and lower boundaries in HSV pixels
#intensities to be considered 'skin'
lower = np.array([0, 30, 60], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

#if video file is not supplied
#use webcam

if not args.get("video", False):
	camera = cv2.VideoCapture(0)

#otherwise load the video
else:
	camera = cv2.VideoCapture(args["video"])

#start reading frames from video
while True:
	#grab the current frame
	(grabbed, frame) = camera.read()

	#if frame is not grabbed during video, video has ended
	if args.get("video") and not grabbed:
		break

	#resize the frame and convert it to hsv
	frame = imutils.resize(frame, width = 400)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)
	#using elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel,  iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	#blur the mask to help remove noise
	#then apply the mask to the frame

	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
	#show the skin in the image along with the mask
	cv2.imshow("images", np.hstack([frame, skin]))

	#if 'q' is pressed , stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

#cleanup the camera
camera.release()
cv2.destroyAllWindows()

