import io

import cv2
import numpy as np
import pytesseract
import pandas as pd

img = cv2.imread('/home/dinesh.krishna@zucisystems.com/workspace/TextEtractionForTable/sample_document_2_0/sample_document_2_0_2.jpg')
image = cv2.resize(img, None, fx=1.9, fy=1.9, interpolation=cv2.INTER_CUBIC)
color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # SET THE GRAY TO IMAGE
thresh = cv2.threshold(color, 120, 255, cv2.THRESH_BINARY)[1]
blur = cv2.medianBlur(thresh, 5)

kernel = np.ones((1, 1), np.uint8)
erode = cv2.erode(blur, kernel, iterations=1)
dilate = cv2.dilate(erode, kernel, iterations=1)
morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

extracted_text = pytesseract.image_to_string(morph)
#print('==================================================')
#extracted_text_data = pytesseract.image_to_data(morph)
#print(extracted_text)

#with open('Extracted_Text.txt', mode='w') as f:
 #   f.write(extracted_text)

#dataframe = pd.read_csv("Extracted_Text.txt",skip_blank_lines=False)

dataframe_direct = pd.read_csv(io.StringIO(extracted_text), skip_blank_lines=False, header=0)
dataframe_direct.drop([0], inplace = True)
#dataframe_direct.drop([1], inplace= True)

print(dataframe_direct)



dataframe_direct.to_csv("Extracted_Text_sample_document_2_0_2.csv",index=True)


cv2.imshow('Morph', morph)
cv2.waitKey()
