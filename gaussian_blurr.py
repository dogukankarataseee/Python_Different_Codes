#HORUS-03-Gaussian-Blur.ipynb
import cv2
import numpy as np

# Set image Size from HORUS waterproof Camera
width   = 1920
height  = 1080


# Create video capture instance
cap = cv2.VideoCapture(0)


## Set up the size of Video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


## Create windo to display the modified video
cv2.namedWindow('mixed_video',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

while(cap.isOpened()):
    # Read one frame from HOURS Camera
    ret, frame = cap.read()
    
    blur = cv2.blur(frame, (5, 5))
    Gaussblur = cv2.GaussianBlur(frame,(5,5),0)

    # display the original frame and modified frame together
    vis = np.concatenate((frame, blur, Gaussblur), axis=1) 
    cv2.imshow("mixed_video", vis);
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()