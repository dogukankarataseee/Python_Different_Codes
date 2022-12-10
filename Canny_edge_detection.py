import cv2 
import numpy as np


# Set image Size from HORUS waterproof Camera
cap = cv2.VideoCapture(0)


# Inform if the camera is availabe
print("HORUS Camera is ready ? {}".format(cap.isOpened()))


#Set up the size of Video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


## Create windo to display the modified video
cv2.namedWindow('Canny-Edge',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)


def nothing(x):
    pass


# Create the RGB bar on the mixed_video window
#Initialize the minVal 
cv2.createTrackbar('minVal', 'Canny-Edge', 0, 255, nothing)
cv2.setTrackbarPos('minVal','Canny-Edge', 100)

#Initialize the maxVal 
cv2.createTrackbar('maxVal', 'Canny-Edge', 0, 255, nothing)
cv2.setTrackbarPos('maxVal','Canny-Edge', 100)


while(True):

    ret, frame = cap.read()

    if not ret:
        break

    Gaussblur = cv2.GaussianBlur(frame,(5,5),0)     
        
    # Read the minVal and maxVal modified by User
    minVal = cv2.getTrackbarPos('minVal','Canny-Edge')
    maxVal = cv2.getTrackbarPos('maxVal','Canny-Edge')      
        
    edges = cv2.Canny(Gaussblur, minVal, maxVal)
    edges_three_channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # display the original frame and modified frame together
    vis = np.concatenate((frame, edges_three_channel), axis=1) 
    cv2.imshow("Canny-Edge", vis);

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()