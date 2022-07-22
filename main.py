import csv

import cv2
import numpy as np
import pytesseract
import pandas as pd

img = cv2.imread('sample_document_1_0/sample_document_1_0_0.jpg')
#print(pytesseract.image_to_string(img))

print('===========================================')
image = cv2.resize(img, None, fx=1.9, fy=1.9, interpolation=cv2.INTER_CUBIC)
print(pytesseract.image_to_string(image))





color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(color, 1)

threashold = cv2.threshold(blur, 127, 225, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((1, 1), np.uint8)
erode = cv2.erode(threashold, kernel, iterations=1)
dilate = cv2.dilate(erode, kernel, iterations=1)
morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
#canny = cv2.Canny(morph, 100, 100)
extracted_text = pytesseract.image_to_string(morph)

with open('file.txt', mode='w') as f:
    f.write(extracted_text)

dataframe = pd.read_csv("file.txt")
dataframe.to_csv('file_csv.csv', index=None)


print("------------------Morph--------------------")
print(pytesseract.image_to_string(morph))
cv2.imshow('Image', morph)
cv2.waitKey(0)

#blur = cv2.GaussianBlur(color, (5, 5), 0)
#blur = cv2.GaussianBlur(color, (0,0), sigmaX=33, sigmaY=33)

# averaging = cv2.boxFilter(image, -1, (5, 5), normalize=True)
# th1 = cv2.threshold(color,127,255,cv2.THRESH_BINARY)



# cv2.imshow('Image', threashold)
# cv2.waitKey(0)


#kernel_morp = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


TextExtraction = pytesseract.image_to_string(morph, lang='eng')
TextExtraction1 = pytesseract.image_to_string(morph, lang='eng')
print(TextExtraction)

#with open('file.txt', mode='w') as f:
 #   f.write(TextExtraction1)

#dataframe1 = pd.read_csv("file.txt")

# storing this dataframe in a csv file
#dataframe1.to_csv('file_csv.csv', index=None)

#dataframes = pd.DataFrame(TextExtraction)
#dataframes.to_csv('file1.csv')

#with open('people.csv', 'w') as outfile:
 #   writer = csv.writer(outfile)
    #writer.replace(",", "")
  #  writer.writerow(TextExtraction1)
#print(dataframes)
