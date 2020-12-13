/**
 * @author: Srinidhi K Prasad
 * @basedon: OpenCV Tutorial, Satya Mallick
 * @description: OCR - Handwritten digits
 * @link: https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
**/

#include <sys/stat.h>
#include <unistd.h>
#include <string>

#define USE_OPENCV_3 

#ifdef USE_OPENCV_3
   //Include libraries
   #include <iostream>
   #include <stdio.h>
   #include <opencv2/opencv.hpp>
   #include <opencv2/highgui.hpp>
   #include <opencv2/imgproc.hpp>
   #include "opencv2/objdetect.hpp"
   #include <opencv2/ml.hpp>

   //Include namespaces
   using namespace cv::ml;
   using namespace std;
   using namespace cv;
#endif

class TrainOCR
{

private:
    //Define and initialize training parameters
    static const string _dataset_path;
    static const string _save_path;  
    static const int _sz = 20; 
    
public:
    TrainOCR();
    ~TrainOCR();
    
    //Define getters for static variables
    static const int getSZ(){ return _sz;}

    static const string& getDatasetPath()
    {
        static const string _dataset_path = "../resource/digits.png";   
        return _dataset_path;
    }
    
    static const string& getSaveModelPath()
    {
        static const string _save_path = "../results/SVMClassifierModel.yml";
        return _save_path;
    }
    
    static Mat Deskew(Mat& img);
    void LoadTrainingData(const string &path, vector <Mat> &trainGrids, vector <Mat> &testGrids, vector <int> &trainLabels, vector <int> &testLabels);
    void LoadDeskewedTrainingData(vector<Mat> &deskewedTrainGrids,vector<Mat> &deskewedTestGrids, vector<Mat> &trainGrids, vector<Mat> &testGrids);
    void LoadHoGTrainingData(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainGrids, vector<Mat> &deskewedtestGrids);
    void VectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat);
    void SVMParams(SVM *SVM);
    void SVMTrain(Ptr<SVM> SVM, Mat &trainMat, vector<int> &trainLabels);
    void SVMPredict(Ptr<SVM> SVM, Mat &testResponse, Mat &testMat);
    void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels);
    void TrainModel(); 
           
    HOGDescriptor HoG
{
    /**
     * @description: Histogram of Oriented Gradients (Feature descriptor)
     * Steps: Crop to feature --> Calculate gradient magnitude/orientation --> Create Histogram
     * Link: https://www.learnopencv.com/histogram-of-oriented-gradients/
     **/

    Size(20,20), //winSize 
    Size(8,8), //blocksize 
    Size(4,4), //blockStride
    Size(8,8), //Gridsize
    9, //nbins
    1, //derivAper
    -1, //winSigma
    0, //histogramNormType
    0.2, //L2HysThresh
    0,  //gamma correction
    64, //nlevels=64
    1
};

};
