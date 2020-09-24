Driver Drowsiness Detection System

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.

The objective of this Python project is to build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected.

In this Python project, we will be using OpenCV for gathering the images from webcam and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The approach we will be using for this Python project is as follows :

Step 1 – Take image as input from a camera.
Step 2 – Detect the face in the image and create a Region of Interest (ROI).
Step 3 – Detect the eyes from ROI and feed it to the classifier.
Step 4 – Classifier will categorize whether eyes are open or closed.
Step 5 – Calculate score to check whether the person is drowsy.

Let’s now understand how our algorithm works step by step.

Step 1 – Take Image as Input from a Camera

With a webcam, we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame. We use the method provided by OpenCV, cv2.VideoCapture(0) to access the camera and set the capture object (cap). cap.read() will read each frame and we store the image in a frame variable.

Step 2 – Detect Face in the Image and Create a Region of Interest (ROI)

To detect the face in the image, we need to first convert the image into grayscale as the OpenCV algorithm for object detection takes gray images in the input. We don’t need color information to detect the objects. We will be using haar cascade classifier to detect faces. This line is used to set our classifier face = cv2.CascadeClassifier(‘ path to our haar cascade xml file’). Then we perform the detection using faces = face.detectMultiScale(gray). It returns an array of detections with x,y coordinates, and height, the width of the boundary box of the object. Now we can iterate over the faces and draw boundary boxes for each face.

for (x,y,w,h) in faces: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 1 )
	
Step 3 – Detect the eyes from ROI and feed it to the classifier

The same procedure to detect faces is used to detect eyes. First, we set the cascade classifier for eyes in leye and reye respectively then detect the eyes using left_eye = leye.detectMultiScale(gray). Now we need to extract only the eyes data from the full image. This can be achieved by extracting the boundary box of the eye and then we can pull out the eye image from the frame with this code.

Step 4 – Classifier will Categorize whether Eyes are Open or Closed
we convert the color image into grayscale using r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY). Then we categorize the eyes are opened or closed.

Step 5 – Calculate Score to Check whether Person is Drowsy

The score is basically a value we will use to determine how long the person has closed his eyes. So if both eyes are closed, we will keep on increasing score and when eyes are open, we decrease the score. We are drawing the result on the screen using cv2.putText() function which will display real time status of the person.

cv2.putText(frame, “Open”, (10, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA )
A threshold is defined for example if score becomes greater than 15 that means the person’s eyes are closed for a long period of time. This is when we beep the alarm using sound.play()

Building the drowsiness detector with OpenCV
To start our implementation, open up a new file, name it detect_drowsiness.py , and insert the following code:


Drowsiness detection with OpenCV
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
Lines 2-12 import our required Python packages.

We’ll need the SciPy package so we can compute the Euclidean distance between facial landmarks points in the eye aspect ratio calculation (not strictly a requirement, but you should have SciPy installed if you intend on doing any work in the computer vision, image processing, or machine learning space).

We’ll also need the imutils package, my series of computer vision and image processing functions to make working with OpenCV easier.

If you don’t already have imutils  installed on your system, you can install/upgrade imutils  via:


Drowsiness detection with OpenCV
$ pip install --upgrade imutils
We’ll also import the Thread  class so we can play our alarm in a separate thread from the main thread to ensure our script doesn’t pause execution while the alarm sounds.

In order to actually play our WAV/MP3 alarm, we need the playsound library, a pure Python, cross-platform implementation for playing simple sounds.

The playsound  library is conveniently installable via pip :


Drowsiness detection with OpenCV
$ pip install playsound
However, if you are using macOS (like I did for this project), you’ll also want to install pyobjc, otherwise you’ll get an error related to AppKit  when you actually try to play the sound:


Drowsiness detection with OpenCV
$ pip install pyobjc
I only tested playsound  on macOS, but according to both the documentation and Taylor Marks (the developer and maintainer of playsound ), the library should work on Linux and Windows as well.

Note: If you are having problems with playsound , please consult their documentation as I am not an expert on audio libraries.

To detect and localize facial landmarks we’ll need the dlib library which is imported on Line 11. If you need help installing dlib on your system, please refer to this tutorial.

Next, we need to define our sound_alarm  function which accepts a path  to an audio file residing on disk and then plays the file:


Drowsiness detection with OpenCV
def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)
We also need to define the eye_aspect_ratio  function which is used to compute the ratio of distances between the vertical eye landmarks and the distances between the horizontal eye landmarks:


Drowsiness detection with OpenCV
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear
The return value of the eye aspect ratio will be approximately constant when the eye is open. The value will then rapid decrease towards zero during a blink.

If the eye is closed, the eye aspect ratio will again remain approximately constant, but will be much smaller than the ratio when the eye is open.

To visualize this, consider the following figure from Soukupová and Čech’s 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks:


Figure 7: Top-left: A visualization of eye landmarks when then the eye is open. Top-right: Eye landmarks when the eye is closed. Bottom: Plotting the eye aspect ratio over time. The dip in the eye aspect ratio indicates a blink (Figure 1 of Soukupová and Čech).
On the top-left we have an eye that is fully open with the eye facial landmarks plotted. Then on the top-right we have an eye that is closed. The bottom then plots the eye aspect ratio over time.

As we can see, the eye aspect ratio is constant (indicating the eye is open), then rapidly drops to zero, then increases again, indicating a blink has taken place.

In our drowsiness detector case, we’ll be monitoring the eye aspect ratio to see if the value falls but does not increase again, thus implying that the person has closed their eyes.

You can read more about blink detection and the eye aspect ratio in my previous post.

Next, let’s parse our command line arguments:


Drowsiness detection with OpenCV
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
Our drowsiness detector requires one command line argument followed by two optional ones, each of which is detailed below:

--shape-predictor : This is the path to dlib’s pre-trained facial landmark detector. You can download the detector along with the source code to this tutorial by using the “Downloads” section at the bottom of this blog post.
--alarm : Here you can optionally specify the path to an input audio file to be used as an alarm.
--webcam : This integer controls the index of your built-in webcam/USB camera.
Now that our command line arguments have been parsed, we need to define a few important variables:


Drowsiness detection with OpenCV
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False
Line 48 defines the EYE_AR_THRESH . If the eye aspect ratio falls below this threshold, we’ll start counting the number of frames the person has closed their eyes for.

If the number of frames the person has closed their eyes in exceeds EYE_AR_CONSEC_FRAMES  (Line 49), we’ll sound an alarm.

Experimentally, I’ve found that an EYE_AR_THRESH  of 0.3  works well in a variety of situations (although you may need to tune it yourself for your own applications).

I’ve also set the EYE_AR_CONSEC_FRAMES  to be 48 , meaning that if a person has closed their eyes for 48 consecutive frames, we’ll play the alarm sound.

You can make the drowsiness detector more sensitive by decreasing the EYE_AR_CONSEC_FRAMES  — similarly, you can make the drowsiness detector less sensitive by increasing it.

Line 53 defines COUNTER , the total number of consecutive frames where the eye aspect ratio is below EYE_AR_THRESH .

If COUNTER  exceeds EYE_AR_CONSEC_FRAMES , then we’ll update the boolean ALARM_ON  (Line 54).

The dlib library ships with a Histogram of Oriented Gradients-based face detector along with a facial landmark predictor — we instantiate both of these in the following code block:

Drowsiness detection with OpenCV
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
The facial landmarks produced by dlib are an indexable list, as I describe here:


Figure 8: Visualizing the 68 facial landmark coordinates from the iBUG 300-W dataset (larger resolution).
Therefore, to extract the eye regions from a set of facial landmarks, we simply need to know the correct array slice indexes:


Drowsiness detection with OpenCV
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
Using these indexes, we’ll easily be able to extract the eye regions via an array slice.

We are now ready to start the core of our drowsiness detector:


Drowsiness detection with OpenCV
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
On Line 69 we instantiate our VideoStream  using the supplied --webcam  index.

We then pause for a second to allow the camera sensor to warm up (Line 70).

On Line 73 we start looping over frames in our video stream.

Line 77 reads the next frame , which we then preprocess by resizing it to have a width of 450 pixels and converting it to grayscale (Lines 78 and 79).

Line 82 applies dlib’s face detector to find and locate the face(s) in the image.

The next step is to apply facial landmark detection to localize each of the important regions of the face:


Drowsiness detection with OpenCV
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
We loop over each of the detected faces on Line 85 — in our implementation (specifically related to driver drowsiness), we assume there is only one face — the driver — but I left this for  loop in here just in case you want to apply the technique to videos with more than one face.

For each of the detected faces, we apply dlib’s facial landmark detector (Line 89) and convert the result to a NumPy array (Line 90).

Using NumPy array slicing we can extract the (x, y)-coordinates of the left and right eye, respectively (Lines 94 and 95).

Given the (x, y)-coordinates for both eyes, we then compute their eye aspect ratios on Line 96 and 97.

Soukupová and Čech recommend averaging both eye aspect ratios together to obtain a better estimation (Line 100).

We can then visualize each of the eye regions on our frame  by using the cv2.drawContours  function below — this is often helpful when we are trying to debug our script and want to ensure that the eyes are being correctly detected and localized:


Drowsiness detection with OpenCV
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
Finally, we are now ready to check to see if the person in our video stream is starting to show symptoms of drowsiness:


Drowsiness detection with OpenCV
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True
					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False
On Line 111 we make a check to see if the eye aspect ratio is below the “blink/closed” eye threshold, EYE_AR_THRESH .

If it is, we increment COUNTER , the total number of consecutive frames where the person has had their eyes closed.

If COUNTER exceeds EYE_AR_CONSEC_FRAMES  (Line 116), then we assume the person is starting to doze off.

Another check is made, this time on Line 118 and 119 to see if the alarm is on — if it’s not, we turn it on.

Lines 124-128 handle playing the alarm sound, provided an --alarm  path was supplied when the script was executed. We take special care to create a separate thread responsible for calling sound_alarm  to ensure that our main program isn’t blocked until the sound finishes playing.

Lines 131 and 132 draw the text DROWSINESS ALERT!  on our frame  — again, this is often helpful for debugging, especially if you are not using the playsound  library.

Finally, Lines 136-138 handle the case where the eye aspect ratio is larger than EYE_AR_THRESH , indicating the eyes are open. If the eyes are open, we reset COUNTER  and ensure the alarm is off.

The final code block in our drowsiness detector handles displaying the output frame  to our screen:


Drowsiness detection with OpenCV
		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
To see our drowsiness detector in action, proceed to the next section.

Testing the OpenCV drowsiness detector
To start, make sure you use the “Downloads” section below to download the source code + dlib’s pre-trained facial landmark predictor + example audio alarm file utilized in today’s blog post.

I would then suggest testing the detect_drowsiness.py  script on your local system in the comfort of your home/office before you start to wire up your car for driver drowsiness detection.

In my case, once I was sufficiently happy with my implementation, I moved my laptop + webcam out to my car (as detailed in the “Rigging my car with a drowsiness detector” section above), and then executed the following command:


Drowsiness detection with OpenCV
$ python detect_drowsiness.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat \
	--alarm alarm.wav
