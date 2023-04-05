import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def main():
    sample_img = cv2.imread('./assets/sample_image.jpg')       
         
    image_height, image_width, _  = sample_img.shape

    detector = faceDetector()
  


    cv2.imwrite(r'./assets/output_result.png', output_image)

    print("Output Image is being displayed")
    cv2.imshow('Output', output_image)                                  #cv2.imshow would give error in google colab so switch to using cv2_imshow by google.colab.patches if using colab
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Output Image window destroyed")


if __name__ == "__main__":
    main()