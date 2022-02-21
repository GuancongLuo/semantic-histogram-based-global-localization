#include "pointCloudMapping.hpp"
#include "neighborhood.hpp"
#include "toolfile.hpp"
#include <string.h>
#include <boost/thread/thread.hpp>
#include "matcher.hpp"
#include "registration.hpp"

#include <pcl/io/pcd_io.h>  //PCD读写类相关的头文件
#include <pcl/point_types.h>    //PCL中支持的点类型头文件
#include "json.hpp" 

using namespace cv;
using namespace std;
using namespace Eigen;

using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <string>


int main(int argc, const char * argv[])
{
    json j; 

    std::string filename = "../src/config.json";

    std::fstream s(filename);
    if (!s.is_open())
    {
        std::cout << "failed to open " << filename << '\n';
    }
    else
    {
        s >> j;
    }

    std::string data_dir = j["Dataset_dir_abs_path"];

    vector<std::string> dir1(2);
    vector<std::string> dir2(2);

    std::string depth_file_name1 = j["pointcloud1"]["depth_file_name"];    
    std::string seg_file_name1 = j["pointcloud1"]["seg_file_name"];
    dir1[0] = data_dir + depth_file_name1;
    dir1[1] = data_dir + seg_file_name1;

    std::string depth_file_name2 = j["pointcloud2"]["depth_file_name"];    
    std::string seg_file_name2 = j["pointcloud2"]["seg_file_name"];
    dir2[0] = data_dir + depth_file_name2;
    dir2[1] = data_dir + seg_file_name2;

    int startPoint1 = j["pointcloud1"]["star_address"];
    int endPoint1 = j["pointcloud1"]["end_address"];
    int startPoint2 = j["pointcloud2"]["star_address"];
    int endPoint2 = j["pointcloud2"]["end_address"];


    // generate the camera parameter (每个数据集都是？)
    vector<float> camera(4);
    float scale = 1;
    camera[0] = j["camera"]["fx"].get<float>(); //fx
    camera[1] = j["camera"]["fy"].get<float>(); //fy
    camera[2] = j["camera"]["cx"].get<float>(); //cx
    camera[3] = j["camera"]["cy"].get<float>(); //cy

    for(float f : camera){
        std::cout <<"camera params: "<< f << std::endl;
    }

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
    // insertPose(dir1,pose1, 4000);
    // insertPose(dir2,pose2, 4000);
    
    //initiallize the intial value of the keypoint;
    vector<vector<float> > centerpoint2;
    vector<vector<float> > centerpoint1;
    
    if (j["is_get_dense_map"])
    {
        gatherDenseMap(cloud1, centerpoint1, pose1, Label, camera, scale, dir1, endPoint1, startPoint1, 1);
        gatherDenseMap(cloud2, centerpoint2, pose2, Label, camera, scale, dir2, endPoint2, startPoint2, 1);

        pcl::io::savePCDFileASCII(j["pointcloud1"]["dense_point_cloud_name"],*cloud1);
        pcl::io::savePCDFileASCII(j["pointcloud2"]["dense_point_cloud_name"],*cloud2);
    }
    else{
        gatherPointCloudData(cloud1, centerpoint1, pose1, Label, label_gray, camera, scale, dir1, endPoint1, startPoint1);
        gatherPointCloudData(cloud2, centerpoint2, pose2, Label, label_gray, camera, scale, dir2, endPoint2, startPoint2);

        pcl::io::savePCDFileASCII(j["pointcloud1"]["point_cloud_name"],*cloud1);
        pcl::io::savePCDFileASCII(j["pointcloud2"]["point_cloud_name"],*cloud2);
    }


    if (j["is_only_get_point_cloud"])
    {
        return 0;
    }

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
