## **Optical Character Recognition**

![Semantic description of image](/source/images/results.png "Optical Character Recognition")

### **Description:**

A simple, C++ project to understand the capabilities of an Optical Character Recognition(OCR) algorithm. This particular implementation, uses the Support Vector Machine(SVM) classifier to train, test and classify digits in an image. 

### **Contents:**

- classification_test.cpp
	- Contains sample code to read, pre-process & recognize digits in a test-image 
- train_ocr.cpp
	- Implements an SVM algorithm that supports training, labelling or testing a hand-written dataset 


### **Dependencies**

* C++ 11 and further
* OpenCV 3.0 
* Cmake 3.5.0 
	
### **Install instructions:**

1. Make sure to meet dependencies.

2. Git clone the project 

```console
user@pc:~$ git clone https://github.com/srinidhi-k-prasad/OCR.git 
```

3. build your project
	
```console
user@pc:~$ OCR/build/cmake..
user@pc:~$ OCR/build/make
```

4. Run project
```console
user@pc:~$ OCR/build/./detect_digit
```

### **Acknowledgements**

Based on the open-source tutorial from Sathya Mallick:

- Handwritten Digits Classification : An OpenCV ( C++ / Python ) Tutorial ([Visit](https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/))
