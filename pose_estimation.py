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

sample_img = cv2.imread('sample_image.jpg')          
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