import cv2
import mediapipe as mp

class poseDetector():

    def __init__(self, mode= True, model_complex=2, smooth_landmarks=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode                                    
        self.model_complex = model_complex
        self.smooth_landmarks = smooth_landmarks              
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.model_complex, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def drawLandmarks(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      
        self.results = self.pose.process(img_RGB)
        # print(type(self.results.pose_landmarks))
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,                    
                                           self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),            
                                           self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)             
                                           )

        return img


def main():
    sample_img = cv2.imread('/assets/sample_image.jpg')       
         
    image_height, image_width, _  = sample_img.shape

    detector = poseDetector()
  
    output_image = detector.drawLandmarks(sample_img)

    print("Output Image is as follows:- ")
    cv2.imshow(output_image)                                  
    cv2.imwrite(r'output_result.png', output_image)


if __name__ == "__main__":
    main()