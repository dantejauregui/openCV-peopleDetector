from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
	help="window stride")
ap.add_argument("-s", "--scale", type=float, default=1.05,
	help="image pyramid scale")
args = vars(ap.parse_args())

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

while True:

	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	frame = imutils.resize(frame, width=400)

	(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()


# USAGE de webcam:
# python detectingCam.py --scale 1.5 --win-stride="(8, 8)"

# USAGE de video mp4:
# python detectingCam.py --video walking2.mp4 --scale 1.5 --win-stride="(8, 8)"
