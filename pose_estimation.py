import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math


def calc_angle(landmark_1, landmark_2, landmark_3):      #Helper fnuction to calculate angle between landmarks
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark_1                      
    x2, y2, _ = landmark_2
    x3, y3, _ = landmark_3       

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))        # Calculating the angle between the three points
    
    if angle < 0:              # if the angle is less than zero add 360 to the found angle.
        angle += 360
    return angle

class poseDetector():
    def __init__(self, mode = True, model_complex=1, smooth_landmarks=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode                                     #static_image_mode           Set as False if wish to work with video frames
        self.model_complex = model_complex
        self.smooth_landmarks = smooth_landmarks              #By default set to True in mpPose.Pose
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.model_complex, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)


    def pose3D(self, img, draw=True, display=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       #Converting BGR to RGB as the mediapipe model expects RGB
        self.results = self.pose.process(img_RGB)
        # print(type(self.results.pose_landmarks))
        output_image = img.copy()
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(output_image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,            #Drawing out the whole skeleton structure
                                            self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),            #Color for landmarks
                                            self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)             #Color for the connection between landmarks
                                            )
        landmarks = []
        image_height, image_width, _  = img.shape

        for landmark in self.results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * image_width), int(landmark.y * image_height),     # Append the landmark into the list.
                                (landmark.z * image_width)))

        if display:
            plt.figure(figsize=[22,22])                                                                       # Displaying the original input image and the resultant image.
            plt.subplot(121);plt.imshow(img[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
            
            self.mpDraw.plot_landmarks(self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)       # Also Plot the Pose landmarks in 3D.
        else:
            return output_image, landmarks          # Return the output image and the found landmarks.

        return img

    def poseClassify(self, landmarks, output_image, draw=True, display=True):
        label = 'UnknownPose'           #Label unknown initially

        # Color (Red) with which the label will be written on the image.
        color = (0, 0, 255)            #Print red colored label if no pose classified
        
        # Calculate the required angles.
        left_elbow_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value], landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value], landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value])
        right_elbow_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value], landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value])           
        left_shoulder_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value], landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value], landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value])
        right_shoulder_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value], landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value])
        left_knee_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value], landmarks[self.mpPose.PoseLandmark.LEFT_KNEE.value], landmarks[self.mpPose.PoseLandmark.LEFT_ANKLE.value])
        right_knee_angle = calc_angle(landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value], landmarks[self.mpPose.PoseLandmark.RIGHT_KNEE.value], landmarks[self.mpPose.PoseLandmark.RIGHT_ANKLE.value])
        
        
        # To check if it is the warrior II pose
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:                   # Check if the both arms are straight.
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:     # Check if shoulders are at the required angle.
                if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:                # Check if one leg is straight.
                    if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:              # Check if the other leg is bended at the required angle.
                        label = 'Warrior II Pose'                                                                                       # Specify the label of the pose that is Warrior II pose.


        # To check if it is the tree pose.
        if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:                        # Check if one leg is straight
            if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:                      # Check if the other leg is bended at the required angle.
                label = 'Tree Pose'
    
        
        if label != 'UnknownPose':       #If the pose is classified successfully
            color = (0, 255, 0)          # Update the color (to green) with which the label will be written on the image.
        
        cv2.putText(output_image, label, (10, 45), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)            # Writing the label on the output image. 
        
        if display:
            plt.figure(figsize=[10,10])
            plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');      
        else:
            return output_image, label


def main():
    sample_img = cv2.imread('./assets/sample_image.jpg')       
         
    image_height, image_width, _  = sample_img.shape

    detector = poseDetector()
  

    output_image, landmarks = detector.pose3D(sample_img, display=False)
    if landmarks:
        detector.poseClassify(landmarks, output_image, display=True)  

    cv2.imwrite(r'./assets/output_result.png', output_image)

    print("Output Image is being displayed")
    cv2.imshow('Output', output_image)                                  #cv2.imshow would give error in google colab so switch to using cv2_imshow by google.colab.patches if using colab
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Output Image window destroyed")

if __name__ == "__main__":
    main()