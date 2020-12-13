/**
 * @author: Srinidhi K Prasad
 * @description: To read an image, recognize digits & reproject results onto image. For function descriptions, refer "TrainOCR.cpp"
**/

#include "classification_test.hpp"

//Create object of class TrainOCR!
TrainOCR trainOCR;

//Create an object of class Classification_Test!
ClassificationTest classificationTest;

ClassificationTest::ClassificationTest()
{

}
ClassificationTest::~ClassificationTest()
{

}

void ClassificationTest::LoadSubGrids(Mat& img, vector <Mat> &sub_grids)
{   
    /**
     * @brief Divide image into sub-grids, each of which holds an image
     * @param img Input image
     * @param sub_grids Output vector of sub-grids
     */

    classificationTest.image = imread(classificationTest.image_path_, cv::IMREAD_GRAYSCALE); 
    int img_count = 0;

    for(int i = 0; i < img.cols; i = i + classificationTest.grid_size_)
    {
        //Clone & Re-size to bring subgrid-image to training convention
        Mat digit_img = ~(img.colRange(i,i+classificationTest.grid_size_)).clone();
        resize(digit_img,digit_img,Size(20,20));   
        
        //Some sub-grid preprocessing to help the SVM-Classifier
        cv::Mat kernel;
        kernel = (cv::Mat_<uchar>(2,2) << 1,0,0,1); 
        erode(digit_img,digit_img, kernel);

        sub_grids.push_back(digit_img); 
        
        //Keep count of the number of subgrids: 81.
        img_count++;
    }
    cout << "Total number of sub-grids detected: " << img_count <<  endl;
}


void ClassificationTest::LoadDeskewedSubGrids(vector<Mat> &deskewed_sub_grids, vector<Mat> &sub_grids)
{   
    /**
     * @brief De-skew image sub-grid if necessary
     * @param deskewed_sub_grids Output vector of de-skewed sub-grids
     * @param sub_grids Input vector of sub-grids
     */

    for(int i=0;i<sub_grids.size();i++)
    {
        Mat deskewed_img = trainOCR.Deskew(sub_grids[i]);
        deskewed_sub_grids.push_back(deskewed_img);

        // //Verify sub-grids after de-skew
        // imshow(to_string(i), deskewed_img);
        // waitKey(0); 
    }  
    cout << "Total number of de-skewed images: " << deskewed_sub_grids.size() <<  endl;
}

void ClassificationTest::HOGCompute(vector<vector<float> > &predict_HoG, vector<Mat> &deskewed_sub_grids)
{   
    /**
     * @brief Compute feature vectors & HoG for sub-grid
     * @param predict_HoG Output 2-D vector: HoG
     * @param deskewed_sub_grids Input vector of de-skewed sub-grids
     */

    for(int y=0;y<deskewed_sub_grids.size();y++)
    {
        vector<float> descriptors;
        trainOCR.HoG.compute(deskewed_sub_grids[y],descriptors);        
        predict_HoG.push_back(descriptors);
    }
    cout << "HoGs Computed!" << endl;
}

void ClassificationTest::VectorToMatrix(int descriptor_size,vector<vector<float> > &predict_HoG,Mat &predict_mat)
{   
    /**
     * @brief Convert 2D-Vector to Matrix
     * @param predict_HoG Input 2D-Vector
     * @param predict_mat Output Matrix
     */

    for(int i = 0;i<predict_HoG.size();i++)
    {
        for(int j = 0;j<descriptor_size;j++)
        {
           predict_mat.at<float>(i,j) = predict_HoG[i][j];
        }
    }
}

void ClassificationTest::StoreToVector(Mat& prediction, vector<int> &sudoku)
{
    for (int i=0; i < prediction.rows; i++)
    {   
        sudoku.push_back(prediction.at<float>(i,0));
    }
}

void ClassificationTest::ReprojectOnImage(Mat& img, vector<int> &sudoku)
{   
    /**
     * @brief Re-project results of prediction on image, and save for later use! ;)
     * @param img Reprojection image
     * @param sudoku  Vector containing results
     */

    int grid_count = 0;
    cvtColor(img,img,COLOR_GRAY2RGB);
    
    for(int i = 0; i < img.cols; i = i + classificationTest.grid_size_)
    {   
        putText(classificationTest.image, to_string(sudoku[grid_count]), Point(i,65), FONT_HERSHEY_DUPLEX, classificationTest.font_scale_, Scalar(0,0,255), classificationTest.font_thickness_);
        imwrite(classificationTest.reprojected_image_, classificationTest.image);
        grid_count ++;
    }
    // //Verify results post-reprojection 
    // imshow("Classification Results", classificationTest.image);
    // waitKey(0); 
}

int main()
{
    //Check if SVM Trained-model is untrained. Otherwise, re-train model.
    Ptr<SVM> SvM = Algorithm::load<SVM>(classificationTest.trained_model_path_);
 
    //Load Trained Model
    if(SvM->isTrained()) 
    {
        cout << "Found OCR Trained Model! No re-training necessary..\n";
    }    
    else 
    {
        cout << "Could not find an existing trained-model. Program will train the OCR-network before character-recognition..\n";
        trainOCR.TrainModel();
        cout << "Training Successful!\n"; 
    }
    
    //Create sub-grids
    vector <Mat> sub_grids;
    classificationTest.LoadSubGrids(classificationTest.image,sub_grids);

    //De-skew sub_grids
    vector<Mat> deskewed_sub_grids;
    classificationTest.LoadDeskewedSubGrids(deskewed_sub_grids, sub_grids);

    //Compute HoG   
    vector<vector<float> > predict_HoG;
    classificationTest.HOGCompute(predict_HoG, deskewed_sub_grids);
    int descriptor_size = SvM->getVarCount();
    Mat predict_mat(predict_HoG.size(),descriptor_size,CV_32FC1);
    classificationTest.VectorToMatrix(descriptor_size, predict_HoG, predict_mat);

    //Predict and store results to a vector
    Mat prediction;
    vector<int> sudoku;
    SvM->predict(predict_mat, prediction);
    classificationTest.StoreToVector(prediction,sudoku);
    
    //Re-project on image to faciliate test and acceptance
    classificationTest.ReprojectOnImage(classificationTest.image,sudoku);
    
    cout << "Classification Complete! Stored to 'Result' folder.."<<endl;

    return 0;
    
}

