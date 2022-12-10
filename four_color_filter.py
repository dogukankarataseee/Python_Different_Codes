#HORUS-02-RGB-SPLIT.ipynb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set image Size from HORUS waterproof Camera
width   = 1920
height  = 1080


# Create video capture instance
cap = cv2.VideoCapture(0)

now = datetime.now()
dt_str = now.strftime("%Y-%m-%d-%H-%M-%S")

# Use XVID encoder
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create VideoWriter
# FPS is 20.0
FileName = 'RGB-SPLIT-'+dt_str+'.avi'
out = cv2.VideoWriter(FileName, fourcc, 20.0, (width*4, height))


## Set up the size of Video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


## Create windo to display the modified video
cv2.namedWindow('RGB-SPLIT',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)


def clearImage(image):
    # Convert the image from BGR to gray
    dark_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    channels = cv2.split(image)

    # Get the maximum value of each channel
    # and get the dark channel of each image
    # record the maximum value of each channel
    a_max_dst = [ float("-inf") ]*len(channels)
    for idx in range(len(channels)):
        a_max_dst[idx] = channels[idx].max()

    dark_image = cv2.min(channels[0],cv2.min(channels[1],channels[2]))

    # Gaussian filtering the dark channel
    dark_image = cv2.GaussianBlur(dark_image,(25,25),0)

    image_t = (255.-0.95*dark_image)/255.
    image_t = cv2.max(image_t,0.5)

    # Calculate t(x) and get the clear image
    for idx in range(len(channels)):
        channels[idx] = cv2.max(cv2.add(cv2.subtract(channels[idx].astype(np.float32), int(a_max_dst[idx]))/image_t,
                                                        int(a_max_dst[idx])),0.0)/int(a_max_dst[idx])*255
        channels[idx] = channels[idx].astype(np.uint8)

    return cv2.merge(channels)


def nothing(x):
    pass


#cv2.createTrackbar('Value of Gamma', 'mixed_video', 100, 1000, nothing)


while(cap.isOpened()):
    # Read one frame from HOURS Camera
    ret, frame = cap.read()

    b, g ,r =cv2.split(frame)

    zeros = np.zeros(frame.shape[:2], dtype = "uint8")          
    merged_r = cv2.merge([zeros, zeros, r])
    merged_g = cv2.merge([zeros, g, zeros])
    merged_b = cv2.merge([b, zeros, zeros])

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
 
    
   # concatenate the result pictures
    vis = np.concatenate((frame, merged_r, merged_g, merged_b), axis=1) 
    #vis = np.concatenate((frame, clrImage), axis=1) 
    cv2.imshow("RGB-SPLIT", vis);
    
    out.write(vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
out.release()
cv2.destroyAllWindows()