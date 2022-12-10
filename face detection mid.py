import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetecion = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, frame, draw=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetecion.process(imgRGB)
        #print(self.results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw ,ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                self.fancyDraw(frame, bbox)
                
                cv2.putText(frame,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
                cv2.imshow('frame', frame)

        return frame, bboxs
    
    def fancyDraw(self, frame, bbox, l = 38, t = 10, rt = 1):
        x, y, w, h, = bbox
        x1, y1 = x + w, y + h
        
        cv2.rectangle(frame, bbox, (255,0,255),1)
        # Top Left x,y 
        cv2.line(frame, (x,y), (x+l,y), (255,0,255), t)
        cv2.line(frame, (x,y), (x,y+l), (255,0,255), t)
        
        # Top Right x1,y 
        cv2.line(frame, (x1,y), (x1-l,y), (255,0,255), t)
        cv2.line(frame, (x1,y), (x1,y+l), (255,0,255), t)
        
        # Bottom Left x,y1 
        cv2.line(frame, (x,y1), (x+l,y1), (255,0,255), t)
        cv2.line(frame, (x,y1), (x,y1-l), (255,0,255), t)
        
        # Bottom Right x1,y1 
        cv2.line(frame, (x1,y1), (x1-l,y1), (255,0,255), t)
        cv2.line(frame, (x1,y1), (x1,y1-l), (255,0,255), t)
        
        
        
        return frame
def main():
    cap= cv2.VideoCapture(0)
    pTime = 0
    detector =FaceDetector()   
    while True:
        ret, frame = cap.read()
        frame,bboxs = detector.findFaces(frame)
        print(bboxs)
        
        cTime =time.time()    
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame,f'FPS: {int(fps)}',(20,80),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)        
        cv2.imshow('frame', frame)
        cv2.waitKey(1) 



















