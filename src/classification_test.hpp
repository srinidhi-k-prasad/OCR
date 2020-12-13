/**
 * @author: Srinidhi K Prasad
 * @description: To classify digits based on the trained model created in "train_ocr.cpp"
**/

#include "train_ocr.hpp"

class ClassificationTest
{

private:
       
public:
    ClassificationTest();
    ~ClassificationTest();

    //Define variables to Classification_Test.cpp
    float font_scale_ {2.5};
    static const int font_thickness_ {2};
    static const int grid_size_ = 46;
    string trained_model_path_ = "../results/SVMClassifierModel.yml";
    string image_path_ = "../resource/sample.png";
    string reprojected_image_ = "../results/reprojected.png";

    //Define matrix to store image
    Mat image;

    //Define functions to predict digits
    bool FileExists(const std::string& filename);
    bool CheckForSavedModel(const std::string& filename);
    void LoadSubGrids(Mat& img, vector <Mat> &subGrids);
    void LoadDeskewedSubGrids(vector<Mat> &deskewedSubGrids, vector<Mat> &SubGrids);
    void HOGCompute(vector<vector<float> > &predictHoG, vector<Mat> &deskewedSubGrids);
    void VectorToMatrix(int descriptor_size,vector<vector<float> > &predictHoG,Mat &predictMat);
    void ReprojectOnImage(Mat& img, vector<int> &Sudoku);
    void StoreToVector(Mat& Prediction, vector<int> &Sudoku);
};
