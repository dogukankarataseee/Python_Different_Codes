#HORUS-01-TuningRGB.ipynb
import cv2
import numpy as np
from datetime import datetime

# Set image Size from HORUS waterproof Camera
width   = 1920
height  = 1080


# Create video capture instance
cap = cv2.VideoCapture(0)

now = datetime.now()
dt_str = now.strftime("%Y-%m-%d-%H-%M-%S")
#print(dt_str)

# Use XVID encoder
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create VideoWriter
# FPS is 20.0
FileName = 'TuningRGB-'+dt_str+'.avi'
out = cv2.VideoWriter(FileName, fourcc, 20.0, (width*2, height))


## Set up the size of Video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


## Create windo to display the modified video
cv2.namedWindow('TuningRGB',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

def nothing(x):
    pass


# Create the RGB bar on the mixed_video window
#Initialize the R factor 
cv2.createTrackbar('RED Factor', 'TuningRGB', 0, 5000, nothing)
cv2.setTrackbarPos('RED Factor','TuningRGB', 1000)

#Initialize the G factor 
cv2.createTrackbar('GREEN Factor', 'TuningRGB', 0, 5000, nothing)
cv2.setTrackbarPos('GREEN Factor','TuningRGB', 1000)

cv2.createTrackbar('BLUE Factor', 'TuningRGB', 0, 5000, nothing)
cv2.setTrackbarPos('BLUE Factor','TuningRGB', 1000)


while(cap.isOpened()):
    # Read one frame from HOURS Camera
    ret, frame = cap.read()
    modified_frame = frame.copy()

    # Read the R G B factor modified by User
    R_Factor = cv2.getTrackbarPos('RED Factor','TuningRGB')
    G_Factor = cv2.getTrackbarPos('GREEN Factor','TuningRGB')
    B_Factor = cv2.getTrackbarPos('BLUE Factor','TuningRGB')

    #Get Time Stamp
    now = datetime.now()
    TimeStamp_str = now.strftime("%Y/%m/%d %H:%M:%S")
    
    #calculate the original RGB mean
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.around(avg_color)
    avg_color_str = str(avg_color) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Original BGR '+avg_color_str, (10,1060), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, TimeStamp_str, (1300,1060), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # modify the frame based on R G B factor from user
    modified_frame[:, :, 2] = (modified_frame[:, :, 2] * (R_Factor/1000)).clip(0,255)
    modified_frame[:, :, 1] = (modified_frame[:, :, 1] * (G_Factor/1000)).clip(0,255) 
    modified_frame[:, :, 0] = (modified_frame[:, :, 0] * (B_Factor/1000)).clip(0,255) 

    m_avg_color_per_row = np.average(modified_frame, axis=0)
    m_avg_color = np.average(m_avg_color_per_row, axis=0)
    m_avg_color = np.around(m_avg_color)
    m_avg_color_str = str(m_avg_color) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(modified_frame, 'BGR '+m_avg_color_str, (10,1060), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    
    # display the original frame and modified frame together
    vis = np.concatenate((frame, modified_frame), axis=1) 
    cv2.imshow("TuningRGB", vis);
    
    out.write(vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
out.release()
cv2.destroyAllWindows()