import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import os


mp_drawing = mp.solutions.drawing_utils             
mp_pose = mp.solutions.pose                         
mp_holistic = mp.solutions.holistic                 
mp_drawstyle = mp.solutions.drawing_styles           


pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)          

sample_img = cv2.imread('./assets/sample_image.jpg')          
image_height, image_width, _  = sample_img.shape          


results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))       


plt.figure(figsize = [10, 10])
plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()


output_image = sample_img.copy()                                     
mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2), mp_pose.POSE_CONNECTIONS)     
cv2.imwrite(r'output_result.png', output_image)

mp_drawing.draw_landmarks(sample_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                           mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),           
                                           mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)        



# cap = cv2.VideoCapture("1.mp4")
cap = cv2.VideoCapture(0)

#For Video input:
prevTime = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    
    extract_landmarks = [0, 11, 12, 13, 14, 25, 26, 27, 28]  #Landmark indices corresponding to [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('BlazePose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()     