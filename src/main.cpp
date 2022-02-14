#include "pointCloudMapping.hpp"
#include "neighborhood.hpp"
#include "toolfile.hpp"
#include <string.h>
#include <boost/thread/thread.hpp>
#include "matcher.hpp"
#include "registration.hpp"

#include <pcl/io/pcd_io.h>  //PCD读写类相关的头文件
#include <pcl/point_types.h>    //PCL中支持的点类型头文件


using namespace cv;
using namespace std;
using namespace Eigen;


int main(int argc, const char * argv[])
{
    if(argc!=7)
    {
        cout << "usage: ./SLAM_Project address frameCount" << endl;
        return 1;
    }

    char file_name1[1024];
    char fullpath1[1024];
    char file_name2[1024];
    char fullpath2[1024];

    sprintf(file_name1, "%s/", argv[1]);
    sprintf(fullpath1,"/Documents/robot_ws/semantic-histogram-based-global-localization/Dataset/%s",file_name1);
    int startPoint1 = atoi(argv[2]);
    int fileNumber1 = atoi(argv[3]);
    cout<<"file number is: "<<fileNumber1<<endl;
    sprintf(file_name2, "%s/", argv[4]);
    sprintf(fullpath2,"/Documents/robot_ws/semantic-histogram-based-global-localization/Dataset/%s",file_name2);
    int startPoint2 = atoi(argv[5]);
    int fileNumber2 = atoi(argv[6]);

    string dir1 = fullpath1;
    string dir2 = fullpath2;
    //generate the camera parameter
    vector<float> camera(4);
    float scale = 1;
    camera[0] = 2066; //fx
    camera[1] = 2120; //fy
    camera[2] = 1354; //cx
    camera[3] = 372;//cy

    //iniltialize the R and T
    MatrixXf R, T;
    R.setIdentity();
    T.setZero();

    //create the value for visual odometry
    Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>);

    Mat pose1 = cv::Mat::zeros(4000, 7, CV_64FC1);
    Mat pose2 = cv::Mat::zeros(4000, 7, CV_64FC1);

    //generate the correspondent rgb value of all labels in the segmentation image
     Mat Label = (cv::Mat_<int>(13, 3)<< 0, 0, 0,
                                    128, 128, 128,
                                    128, 0, 0,
                                    128, 64, 128,
                                    0, 0, 192,
                                    64, 64, 0,
                                    128,128,0,
                                    192, 192, 128,
                                    64, 0, 128,
                                    192, 128, 128,
                                    64, 64, 0,
                                    0, 128, 192,
                                    0,175,0);
    //Obtain the label number
    vector<uchar> label_gray = GetLabelGray(Label);

    //insert the odeometry value
    insertPose(dir1,pose1, 4000);
    insertPose(dir2,pose2, 4000);
    
    //initiallize the intial value of the keypoint;
    vector<vector<float> > centerpoint2;
    vector<vector<float> > centerpoint1;
    
    gatherPointCloudData(cloud1, centerpoint1, pose1, Label, label_gray, camera, scale, dir1, fileNumber1, startPoint1);
    gatherPointCloudData(cloud2, centerpoint2, pose2, Label, label_gray, camera, scale, dir2, fileNumber2, startPoint2);


    pcl::io::savePCDFileASCII("cloud1_3500_dense.pcd",*cloud1);
    pcl::io::savePCDFileASCII("cloud2_3700_dense.pcd",*cloud2);

    return 0;

    //add the edge between the neighborhood
    Neighborhood Nei1(centerpoint1);

    //MatrixXf neighbor;
    Neighborhood Nei2(centerpoint2);
    //neighbor = Nei2.getNeighbor();
    //cout<<neighbor<<endl;

    //obtain the descriptor
    MatrixXf descriptor1;
    Descriptor Des1(Nei1, 1);
    //descriptor1 = Des1.getDescriptor(4);
    //cout<<descriptor1<<endl;

    MatrixXf descriptor2;
    Descriptor Des2(Nei2, 1);
    // descriptor2 = Des2.getDescriptor(23);
    // cout<<descriptor2<<endl;
    
    //matching
    MatrixXi matcherID;
    matcher matches(Des1, Des2, 2);
    matcherID = matches.getGoodMatcher();
    //cout<<matcherID<<endl;

    //begain registration
    //insert the value
    registration registration(centerpoint1, centerpoint2, matcherID);

    //reject the outliers with ICP-RANSAC method
    registration.matcherRANSAC(10);
    MatrixXi inlierID;
    inlierID = registration.inlierID;

    //final pose estiamation
    registration.Alignment();
    
    R = registration.Rotation;
    T = registration.Translation;
    
    // float finalDistance = sqrt(pow(T(0, 0), 2) + pow(T(1, 0),2) + pow(T(2, 0),2));
    
    // cout<<"final distance: "<<finalDistance<<endl;


    // //plot the semantic point and matching with PCL library  
    pointCloudMapping pointCloudMapping;
    pointCloudMapping.pointVisuallize(cloud1, cloud2, inlierID, R, T);
    
}
