import io
import os
from os import listdir

import cv2
import numpy as np
import pandas as pd
from pytesseract import pytesseract

folder_dir = "/home/dinesh.krishna@zucisystems.com/workspace/TextEtractionForTable/TextExtraction/sample_document_2_0"

extracted_list=[]
for images in os.listdir(folder_dir):
    dir = os.path.join(folder_dir,images)
    #print(dir)
    #print(images)
    if (images.endswith(".jpg")):
        try:
            img = cv2.imread(dir)
            #print(img)
            #print(img)
            image = cv2.resize(img, None, fx=1.9, fy=1.9, interpolation=cv2.INTER_CUBIC)

            #print(cv2.shape(image))
            color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # SET THE GRAY TO IMAGE
            thresh = cv2.threshold(color, 120, 255, cv2.THRESH_BINARY)[1]
            blur = cv2.medianBlur(thresh, 5)

            kernel = np.ones((1, 1), np.uint8)
            erode = cv2.erode(blur, kernel, iterations=1)
            dilate = cv2.dilate(erode, kernel, iterations=1)
            morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
            print(image.shape)
            canny=cv2.Canny(morph,50,50)
            imgLines = cv2.HoughLinesP(canny, 15, np.pi / 180, 10)

            print(imgLines)

            #imS = cv2.resize(morph, (300, 600))  # Resize image
            #cv2.imshow("output", imgLines)  # Show image
            #cv2.waitKey(0)
            x, y, w, h = 0, 460, 1130, 308
          #  x,y=33,198


            ROI = morph[y:y+h,x:x+w]
            print(ROI)
            #cv2.imshow('Image', morph)
            #cv2.waitKey(0)
            # crop=morph[210:2000,0:4000]
            cv2.imshow('crop',ROI)
            cv2.waitKey(0)

            extracted_text = pytesseract.image_to_string(ROI,lang='eng')
            print(extracted_text)
            dataframe_direct = pd.read_csv(io.StringIO(extracted_text), skip_blank_lines=True,header=0)
            print(dataframe_direct)
            extracted_list.append(dataframe_direct)
            df=pd.concat(extracted_list,axis=1)
            pd.options.display.max_columns = None
            #df = df.set_index('id_column')
            #df=pd.DataFrame(extracted_list)
            #print('length'+len(data_frame))
            #print('Count'+data_frame.count())
            #print(data_frame)
            #print(extracte_list)

            df.to_csv('Extracted Text.csv',index=True)

        except Exception as e:
            print(e)






