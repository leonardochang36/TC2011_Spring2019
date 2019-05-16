//
//  main.cpp
//  OpenVC
//
//  Created by Alejandro Herce on 2/28/19.
//  Copyright © 2019 Alejandro Herce. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "drawLandmarks.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        cout << "err" << endl;
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            cout << path << endl;
            images.push_back(imread(path, 1));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void authenticate() {
    // Load Face Detector
    CascadeClassifier faceDetector("/Users/herce/Documents/Tec/Sistemas Inteligentes/OpenVC/OpenVC/haarcascade_frontalface_alt2.xml");
    
    // Set up webcam for video capture
    VideoCapture cam;
    
    cam.open(0);
    
    // Variable to store a video frame and its grayscale
    Mat frame, gray, crop, resizedTmp, resizedGrey;
    Rect cropped;
    
    vector<Mat> images, resizedImgs;
    vector<int> labels;
    
    int count = 0;
    
    if (cam.isOpened()) {

        while(cam.read(frame)) {
            
            imshow("Cam", frame);
            waitKey(5);
            // Find face
            vector<Rect> faces;
            // Convert frame to grayscale because
            // faceDetector requires grayscale image.
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            
            // Detect faces
            faceDetector.detectMultiScale(gray, faces);
            
            if (faces.size() > 0 && count > 60) {
                
                cropped.x = faces[0].x;
                cropped.y = faces[0].y;
                cropped.width = faces[0].width;
                cropped.height = faces[0].height;
                
                crop = frame(cropped);
                
                imshow("Face", crop);
                
                images.push_back(crop.clone());
                labels.push_back(count);
                
                cout << "face frame saved" << endl;
            }
            
            count++;
            
            if (images.size() == 25) break;
        }
        
        cout << images.size() << endl;
        
        cout << "resizing images" << endl;
        
        for (int i = 0; i < images.size(); i++) {
            
            try {
                resize(images[i], resizedTmp, Size(296,296));
            } catch (cv::Exception& e) {
                cerr << "Err file " << i << endl;
            }
            
            if (resizedTmp.channels() == 3) {
                cvtColor(resizedTmp, resizedGrey, COLOR_BGR2GRAY);
            }
            if (resizedTmp.channels() == 4) {
                cvtColor(resizedTmp, resizedGrey, COLOR_BGRA2GRAY);
            }
            
            resizedImgs.push_back(resizedGrey.clone());
            // imwrite("test-" + to_string(i) + ".jpg", resizedGrey);
        }
        
        cout << "magic stuff happening" << endl;
        
        Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
        
        model->read("eigenfaces.yml");
        
        for(int i = 0; i < resizedImgs.size(); i++) {
            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(resizedImgs[i], predictedLabel, confidence);
            cout << "Predicted label: " << predictedLabel << " confidence: " << confidence << " for: " << i << endl;
        }
        
        Mat eigenvalues = model->getEigenValues();
        // And we can do the same to display the Eigenvectors (read Eigenfaces):
        Mat W = model->getEigenVectors();
        // Get the sample mean from the training data
        Mat mean = model->getMean();
        // Display or save:
        imshow("mean", norm_0_255(mean.reshape(1, resizedImgs[0].rows)));
        
    } else {
        cout << "Error opening camera" << endl;
    }
}

void saveFaces(bool isRealFace) {
    // Load Face Detector
    CascadeClassifier faceDetector("/Users/herce/Documents/Tec/Sistemas Inteligentes/OpenVC/OpenVC/haarcascade_frontalface_alt2.xml");
    
    // Set up webcam for video capture
    VideoCapture cam;
    
    cam.open(0);
    
    // Variable to store a video frame and its grayscale
    Mat frame, gray, crop, resizedTmp, resizedGrey;
    Rect cropped;
    
    vector<Mat> images, resizedImgs;
    vector<int> labels;
    
    int count = 0;
    
    if (cam.isOpened()) {
        while(cam.read(frame)) {
            // Find face
            vector<Rect> faces;
            // Convert frame to grayscale because
            // faceDetector requires grayscale image.
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            
            // Detect faces
            faceDetector.detectMultiScale(gray, faces);
            
            if (faces.size() > 0) {
                
                cropped.x = faces[0].x;
                cropped.y = faces[0].y;
                cropped.width = faces[0].width;
                cropped.height = faces[0].height;
                
                crop = frame(cropped);
                
                imshow("Face", crop);
                
                images.push_back(crop.clone());
                labels.push_back(count);
                
                if (waitKey(0) == 27) {
                    if (isRealFace) {
                        imwrite("real/real-" + to_string(count) + ".jpg", crop);
                    } else {
                        imwrite("fake/fake-" + to_string(count) + ".jpg", crop);
                    }
                }
                count++;
            }
            
            if (images.size() == 120) break;
        }
    } else {
        cout << "Error opening camera" << endl;
    }
}

void trainModel() {
    // Variable to store a video frame and its grayscale
    Mat frame, gray, crop, resizedTmp, resizedGrey;
    Rect cropped;
    
    vector<Mat> images, resizedImgs;
    vector<int> labels;
    
    try {
        read_csv("pics.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file. Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        cerr << error_message << endl;
    }
    
    cout << images.size() << endl;
    
    for (int i = 0; i < images.size(); i++) {
        
        cout << "resizing " << i << endl;
        
        try {
            resize(images[i], resizedTmp, Size(296,296));
        } catch (cv::Exception& e) {
            cerr << "Err file " << i << endl;
        }
        
        cout << resizedTmp.channels() << endl;
        
        if (resizedTmp.channels() == 3) {
            cvtColor(resizedTmp, resizedGrey, COLOR_BGR2GRAY);
        }
        if (resizedTmp.channels() == 4) {
            cvtColor(resizedTmp, resizedGrey, COLOR_BGRA2GRAY);
        }
        
        resizedImgs.push_back(resizedGrey.clone());
        // imwrite("test-" + to_string(i) + ".jpg", resizedGrey);
    }
    
    cout << "magic stuff happening" << endl;
    
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    
    model->train(resizedImgs, labels);
    model->save("eigenfaces.yml");
    
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save:
    imshow("mean", norm_0_255(mean.reshape(1, resizedImgs[0].rows)));
    
    cout << "Model trained!" << endl;
}

int main(int argc, const char * argv[]) {
    
    int choice = 0;
    
    while (choice < 5) {
        cout << "*******************************" << endl;
        cout << " 1 - Authenticate with trained model" << endl;
        cout << " 2 - Save real faces in computer" << endl;
        cout << " 3 - Save fake faces in computer" << endl;
        cout << " 4 - Train model with saved faces" << endl;
        cout << " 5 - Exit" << endl;
        cout << " Enter your choice and press return: ";
        
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << "Authenticate" << endl;
                authenticate();
                break;
            case 2:
                cout << "Save real faces" << endl;
                saveFaces(true);
                break;
            case 3:
                cout << "Save fake faces" << endl;
                saveFaces(false);
                break;
            case 4:
                cout << "Train model" << endl;
                trainModel();
                break;
            case 5:
                cout << "End of Program" << endl;
                break;
            default:
                cout << "Not a Valid Choice. \n";
                cout << "Choose again.\n";
                cin >> choice;
                break;
        }
    }
}
