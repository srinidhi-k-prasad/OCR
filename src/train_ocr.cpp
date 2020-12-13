#include "train_ocr.hpp"

TrainOCR::TrainOCR()
{   
    //Train, Test, Evaluate and Return Results!
}

TrainOCR::~TrainOCR()
{ 
    //Destruct, I will
}


Mat TrainOCR::Deskew(Mat& img)
{
   Moments m = moments(img);    
    if(abs(m.mu02)< 1e-2)
    { 
        return img.clone(); 
    }
    
    double skew = m.mu11/m.mu02;
    Mat warp_mat = (Mat_<double>(2,3) << 1, skew, -0.5*TrainOCR::getSZ()*skew, 0, 1 , 0);    
    Mat img_out = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, img_out, warp_mat, img_out.size(), WARP_INVERSE_MAP|INTER_LINEAR);
    return img;
} 

void TrainOCR::LoadTrainingData(const string &path, vector <Mat> &train_grids, vector <Mat> &test_grids, vector <int> &train_labels, vector <int> &test_labels)
{   
    //Load train and test images and labels
    Mat img = imread(path, IMREAD_GRAYSCALE);
    int img_count = 0;

    for(int i = 0; i < img.rows; i = i + TrainOCR::getSZ())
    {
        for(int j = 0; j < img.cols; j = j + TrainOCR::getSZ())
        {   
            //Read the 20x20 pixel digit image
            Mat digit_img = (img.colRange(j,j+TrainOCR::getSZ()).rowRange(i,i+TrainOCR::getSZ())).clone();
            
            //Add all digit-images in 90% (450-rows) of the columns
            if(j < int(0.9*img.cols)) 
            {
                train_grids.push_back(digit_img);
            }
            else
            {
                test_grids.push_back(digit_img);
            }
            img_count++;
        }
    }
    
    cout << "Total number of Images considered: " << img_count << endl;
    float digit_class_ID = 0;

    //Classify images into digits with number classes
    for(int z=0;z<int(0.9*img_count);z++){
        if(z % 450 == 0 && z != 0){ 
            digit_class_ID = digit_class_ID + 1;
            }
        train_labels.push_back(digit_class_ID);
    }
    digit_class_ID = 0;
    for(int z=0;z<int(0.1*img_count);z++){
        if(z % 50 == 0 && z != 0){
            digit_class_ID = digit_class_ID + 1;
            }
        test_labels.push_back(digit_class_ID);
    }
}

void TrainOCR::LoadDeskewedTrainingData(vector<Mat> &deskewed_train_grids,vector<Mat> &deskewed_test_grids, vector<Mat> &train_grids, vector<Mat> &test_grids)
{   
    //load de-skewed training data
    for(int i=0;i<train_grids.size();i++){

        Mat deskewed_img = Deskew(train_grids[i]);
        deskewed_train_grids.push_back(deskewed_img);
    }

    for(int i=0;i<test_grids.size();i++){

        Mat deskewed_img = Deskew(test_grids[i]);
        deskewed_test_grids.push_back(deskewed_img);
    }    
}

void TrainOCR::LoadHoGTrainingData(vector<vector<float> > &train_HOG, vector<vector<float> > &test_HOG, vector<Mat> &deskewed_train_grids, vector<Mat> &deskewed_test_grids)
{   
    //Compute HoGs
    for(int y=0;y<deskewed_train_grids.size();y++){
        vector<float> descriptors;
        
        TrainOCR::HoG.compute(deskewed_train_grids[y],descriptors);
        train_HOG.push_back(descriptors);
    }

    for(int y=0;y<deskewed_test_grids.size();y++){

        vector<float> descriptors;
        TrainOCR::HoG.compute(deskewed_test_grids[y],descriptors);
        test_HOG.push_back(descriptors);
    }
}

void TrainOCR::VectortoMatrix(vector<vector<float> > &train_HOG, vector<vector<float> > &test_HOG, Mat &train_mat, Mat &test_mat)
{   
    int descriptor_size = train_HOG[0].size();
    for(int i = 0;i<train_HOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           train_mat.at<float>(i,j) = train_HOG[i][j];
        }
    }
    for(int i = 0;i<test_HOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            test_mat.at<float>(i,j) = test_HOG[i][j];
        }
    }
}

void TrainOCR::SVMParams(SVM *SVM)
{
    /**
    * @Algortihm: Support Vector Machines (SVM)
    * @description: SVM divides space using planes based on class labels.
    * @param: C - Choose what provides the best classification on a held out test set.
    * **/

    cout << "Kernel type     : " << SVM->getKernelType() << endl;
    cout << "Type            : " << SVM->getType() << endl;
    cout << "C               : " << SVM->getC() << endl;
    cout << "Degree          : " << SVM->getDegree() << endl;
    cout << "Nu              : " << SVM->getNu() << endl;
    cout << "Gamma           : " << SVM->getGamma() << endl;
}

Ptr<SVM> SVMInit(float C, float gamma)
{
  Ptr<SVM> SVM = SVM::create();
  SVM->setGamma(gamma);
  SVM->setC(C);
  SVM->setKernel(SVM::RBF);
  SVM->setType(SVM::C_SVC);

  return SVM;
}

void TrainOCR::SVMTrain(Ptr<SVM> SVM, Mat &train_mat, vector<int> &train_labels)
{
  Ptr<TrainData> training_data = TrainData::create(train_mat, ROW_SAMPLE, train_labels);
  SVM->train(training_data);
  SVM->save(TrainOCR::getSaveModelPath());
}

void TrainOCR::SVMPredict(Ptr<SVM> SVM, Mat &test_response, Mat &test_mat)
{
  SVM->predict(test_mat, test_response);
}

void TrainOCR::SVMevaluate(Mat &test_response, float &count, float &accuracy, vector<int> &test_labels)
{
  for(int i = 0; i < test_response.rows; i++)
  {
    if(test_response.at<float>(i,0) == test_labels[i])
      count = count + 1;
  }
  accuracy = (count/test_response.rows)*100;
}

void TrainOCR::TrainModel() 
{
  
    vector <Mat> train_grids;
    vector <Mat> test_grids;
    vector <int> train_labels;
    vector <int> test_labels;

    // No data- No training!
    LoadTrainingData(TrainOCR::getDatasetPath(),train_grids,test_grids,train_labels,test_labels);

    //de-skew images and load training data!
    vector<Mat> deskewed_train_grids;
    vector<Mat> deskewed_test_grids;
    LoadDeskewedTrainingData(deskewed_train_grids,deskewed_test_grids,train_grids,test_grids);
    
    //Compute HoG and obtain descriptor!
    vector<vector<float> > train_HOG;
    vector<vector<float> > test_HOG;
    LoadHoGTrainingData(train_HOG,test_HOG,deskewed_train_grids,deskewed_test_grids);  
    int descriptor_size = train_HOG[0].size();
    cout << "Descriptor Size : " << descriptor_size << endl;

    //Convert 2-D Vector to a Matrix    
    Mat train_mat(train_HOG.size(),descriptor_size,CV_32FC1);
    Mat test_mat(test_HOG.size(),descriptor_size,CV_32FC1);
    VectortoMatrix(train_HOG,test_HOG,train_mat,test_mat);
    
    //Init, Train, Test, Predict and Evaluate. 
    Mat test_response;
    float C = 12.5;
    float gamma = 0.5;
    
    Ptr<SVM> model = SVMInit(C, gamma);
    SVMTrain(model, train_mat, train_labels);
    SVMPredict(model, test_response, test_mat);

    float count = 0;
    float accuracy = 0 ;
    SVMParams(model);
    SVMevaluate(test_response, count, accuracy, test_labels);

    cout << "Accuracy :" << accuracy << endl;
}