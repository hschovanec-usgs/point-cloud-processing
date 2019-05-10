#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>

// PCL subsampling
#include <pcl/filters/voxel_grid.h>

// PCL IO 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Normals
#include <pcl/features/normal_3d_omp.h>


// Boost property tree
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// Registration
#include <pcl/features/fpfh_omp.h>
#include <pcl/Keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <boost/log/core.hpp>

using namespace std;
using namespace pcl;
using namespace io;
using namespace boost;

property_tree::ptree CONFIG;

struct PointXYZRGBI
{
  PCL_ADD_POINT4D;  
  float r;
  float g;
  float b;
  float i;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBI,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, r, r)
                                   (float, g, g)
                                   (float, b, b)
                                   (float, i, i)
)


// Function headers
PointCloud<PointXYZ>::Ptr CalculateKeypoints(PointCloud<PointXYZ>::Ptr &cloud);
PointCloud<Normal>::Ptr CalculateNormals(PointCloud<PointXYZ>::Ptr &cloud, const float radius);
double CalculateResolution(const PointCloud<PointXYZ>::ConstPtr& cloud);
PointCloud<PointXYZ>::Ptr CopyCloud(PointCloud<PointXYZ>::Ptr input);
void FeatureAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, float inlier, float feature,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud);
void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud);
PointCloud<PointXYZ>::Ptr LoadAsciiCloud(string filepath);
PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz);

bool
loadCloud (const string &filename, PointCloud<PointXYZRGBI> &cloud, PointCloud<PointXYZ> &simpleCloud)
{
  ifstream fs;
  fs.open (filename.c_str (), ios::binary);
  if (!fs.is_open () || fs.fail ())
  {
    cout << "Could not open file." << endl; 
    fs.close ();
    return (false);
  }
  
  string line;
  vector<string> st;

  while (!fs.eof ())
  {
    getline (fs, line);
    // Ignore empty lines
    if (line == "")
      continue;

    // Tokenize the line
    boost::trim (line);
    boost::split (st, line, boost::is_any_of ("\t\r, "), boost::token_compress_on);
	float r, g, b, i;
    if (st.size () < 3){
		continue;
	}
	else if (st.size () == 3){
		r = 0;
		g = 0;
		b = 0;
		i = 0;
	}
	else if (st.size () == 4){
		r = 0;
		g = 0;
		b = 0;
		i = float (atof (st[3].c_str ()));
	}
	else if (st.size () == 6){
		r = float (atof (st[3].c_str ()));
		g = float (atof (st[3].c_str ()));
		b = float (atof (st[3].c_str ()));
		i = 0;
	}
	else if (st.size () >= 7){
		r = float (atof (st[4].c_str ()));
		g = float (atof (st[5].c_str ()));
		b = float (atof (st[6].c_str ()));
		i = float (atof (st[3].c_str ()));
	}
	PointXYZRGBI point;
	point.x = float (atof (st[0].c_str ()));
	point.y = float (atof (st[1].c_str ()));
	point.z = float (atof (st[2].c_str ()));
	point.r = r;
	point.g = g;
	point.b = b;
	point.i = i;
    cloud.push_back (point);

	PointXYZ simplePoint;
	simplePoint.x = float (atof (st[0].c_str ()));
	simplePoint.y = float (atof (st[1].c_str ()));
	simplePoint.z = float (atof (st[2].c_str ()));
	simpleCloud.push_back(simplePoint);
  }
  fs.close ();

  cloud.width = uint32_t (cloud.size ()); cloud.height = 1; cloud.is_dense = true;
  return (true);
}


// Inline functions
float deg2rad(float deg){
	float rad = (deg * 3.14159265359) / 180;
	return rad;
}


property_tree::ptree LoadConfig(string configFile){
	property_tree::ptree pt;
    property_tree::read_json(configFile, pt);
	return pt;
}

void Log(string message){

	return;
}

void SetupLogger(){
	//log::add_file_log("sample.log");
	return;
}





int main (int argc, char** argv){

	string configFile = argv[1];
	CONFIG = LoadConfig(configFile);
	SetupLogger();
	

	PCDWriter w;
	// Load reference cloud
	PointCloud<PointXYZRGBI>::Ptr cloud1 (new PointCloud<PointXYZRGBI>);
	PointCloud<PointXYZ>::Ptr simpleCloud1 (new PointCloud<PointXYZ>);
	cout << "Loading reference cloud" << endl;
	if (!loadCloud (baselineFile, *cloud1, *simpleCloud1)){
		cout << "Unable to load file: " << baselineFile << endl;
		return (-1);
	}

	// Load reference cloud
	PointCloud<PointXYZRGBI>::Ptr mergedCloud (new PointCloud<PointXYZRGBI>);
	   

	for (int i=1; i <= numFiles; i++){
		long long ii = i;
		string suffix = to_string(ii);

	   // Loading cloud to align
	   PointCloud<PointXYZ>::Ptr simpleCloud2 (new PointCloud<PointXYZ>);
	   PointCloud<PointXYZRGBI>::Ptr cloud2 (new PointCloud<PointXYZRGBI>);
	   cout << "Loading cloud to align" << endl;
	   if (!loadCloud (cloudFile, *cloud2, *simpleCloud2)){
		   cout << "Unable to load file: " << cloudFile << endl;
		   return (-1);
	   }
	   cout << "Clouds successfully loaded." << endl;
	   string ftext = cloudFile + "_" + to_string((long long)i) + "_aligned.pcd";

	   double resolution = CalculateResolution(simpleCloud2);
   
	   float lx = 8 * resolution;
	   float radius = 5 * lx;
	   float inlier = .25;
	   float feature = radius * 2;
	   cout << "multiplier " << resolution/lx << endl;

		PointCloud<PointXYZ>::Ptr filteredCloud (new PointCloud<PointXYZ>);
		PointCloud<PointXYZ>::Ptr filteredReference (new PointCloud<PointXYZ>);
	
		*filteredReference = *simpleCloud1;
		*filteredCloud = *simpleCloud2;

		// Perform the voxel filtering
		filteredReference = VoxelFilter(simpleCloud1, filteredReference, lx, lx, lx);
		filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, lx, lx, lx);

		// Calculate normals
		cout << "Calculate Normals" << endl;
		PointCloud<Normal>::Ptr normalReference = CalculateNormals(filteredReference, radius);
		PointCloud<Normal>::Ptr normalCloud = CalculateNormals(filteredCloud, radius);

	   cout << "Starting feature-based alignment." << endl;
	   FeatureAlign(cloud2, inlier, feature, filteredReference, filteredCloud, normalReference, normalCloud);
	   int start = 10;
	   for (int sc=3; sc>0; sc--){
		 int scale = start*sc;
	   // Perform the voxel filtering
		filteredReference = VoxelFilter(simpleCloud1, filteredReference, scale * resolution, scale * resolution, scale * resolution);
		filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, scale * resolution, scale * resolution, scale * resolution);

		// Calculate normals
		cout << "Calculate Normals" << endl;
		normalReference = CalculateNormals(filteredReference, 6*scale*resolution);
		normalCloud = CalculateNormals(filteredCloud, 6*scale* resolution);

		cout << "Starting iterative alignment." << endl;
		IterativeAlign(cloud2, filteredReference, filteredCloud, normalReference, normalCloud);
	   }
		w.writeBinaryCompressed (ftext, *cloud2);
		if (i==1){
			mergedCloud->points = cloud2->points;
		}else{
			mergedCloud->points.insert(mergedCloud->points.end(), cloud2->points.begin(), cloud2->points.end());
		}
	}
    w.writeBinaryCompressed (cloudFile + "_merged.pcd", *mergedCloud);
	
   return (0);
}

PointCloud<PointXYZ>::Ptr CalculateKeypoints(PointCloud<PointXYZ>::Ptr &cloud)
{
	PointCloud<PointXYZ>::Ptr keypoints (new PointCloud<PointXYZ>());
	search::KdTree<PointXYZ>::Ptr tree = search::KdTree<PointXYZ>::Ptr (new search::KdTree<PointXYZ>);

	ISSKeypoint3D<PointXYZ, PointXYZ> keypointDetector;
	keypointDetector.setSalientRadius (0.5);
	keypointDetector.setNonMaxRadius (1);
	keypointDetector.setInputCloud (cloud);
	keypointDetector.setSearchMethod(tree);

	double resolution = CalculateResolution(cloud);
	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	keypointDetector.setSalientRadius(5*resolution);
	//cout << resolution << endl;
	// Set the radius for the application of the non maxima supression algorithm.
	keypointDetector.setNonMaxRadius(3*resolution);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	keypointDetector.setMinNeighbors(5);
	// Set the upper bound on the ratio between the second and the first eigenvalue.
	keypointDetector.setThreshold21(0.975);
	// Set the upper bound on the ratio between the third and the second eigenvalue.
	keypointDetector.setThreshold32(0.975);
	// Set the number of prpcessing threads to use. 0 sets it to automatic.
	keypointDetector.setNumberOfThreads(8);
	keypointDetector.compute(*keypoints);

	return keypoints;
 }

PointCloud<Normal>::Ptr CalculateNormals(PointCloud<PointXYZ>::Ptr &cloud, const float radius)
 {
	 // Create instance of the normal estimation class
	 NormalEstimationOMP<PointXYZ, Normal> normalEstimation;
	 normalEstimation.setViewPoint(numeric_limits<float>::max (), numeric_limits<float>::max (), numeric_limits<float>::max ());
	 normalEstimation.setInputCloud (cloud);
	 cout << "Numerical Limites: " << numeric_limits<float>::max ();
	 
	 // An empty kdtree is required for searching
	 search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ> ());
	 cout << "Set tree: " << endl;
	 normalEstimation.setSearchMethod (tree);
	 // Output datasets
	 PointCloud<Normal>::Ptr cloudNormals (new PointCloud<Normal>);
	 // Use all neighbors in a sphere of radius RAD
	 cout << "Set radius: " << endl;
	 normalEstimation.setRadiusSearch (radius);
	 cout << "Set Threads: " << endl;
	 normalEstimation.setNumberOfThreads(8);
	 cout << "Compute: " << endl;
	 normalEstimation.compute (*cloudNormals);
	 return cloudNormals;
 }

double CalculateResolution(const PointCloud<PointXYZ>::ConstPtr& cloud){
	double resolution = 0.0;
	int numberOfPoints = 0;
	int numberOfNeighbors = 0;

	vector<int> indices(2);
	vector<float> squaredDistances(2);
	search::KdTree<PointXYZ> tree;
	tree.setInputCloud(cloud);

	// Calculate the distance to the nearest neighbor for all points in the cloud and mean
	for (int i = 0; i < cloud->size(); ++i)
	{
		if (! pcl_isfinite((*cloud)[i].x))
			continue;

		// Consider the nearest neighbor not including the point itself
		numberOfNeighbors = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (numberOfNeighbors == 2)
		{
				resolution += sqrt(squaredDistances[1]);
				++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;

	cout << "Resolution: " << resolution << endl;

	return resolution;
}

double CalculateMeanDistance(const PointCloud<PointXYZ>::ConstPtr& cloud){
	double distance = 0.0;
	int numberOfPoints = 0;
	int numberOfNeighbors = 0;

	vector<int> indices(8);
	vector<float> squaredDistances(8);
	search::KdTree<PointXYZ> tree;
	tree.setInputCloud(cloud);

	// Calculate the distance to the nearest neighbor for all points in the cloud and mean
	for (int i = 0; i < cloud->size(); ++i)
	{
		if (! pcl_isfinite((*cloud)[i].x))
			continue;

		// Consider the nearest neighbor not including the point itself
		numberOfNeighbors = tree.nearestKSearch(i, 8, indices, squaredDistances);
		if (numberOfNeighbors == 8)
		{
			for (int i = 1; i < 8; i++){
				distance += squaredDistances[i];
				++numberOfPoints;
			}
		}
	}
	if (numberOfPoints != 0)
		distance /= numberOfPoints;

	return distance;
}

void FeatureAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, float inlier, float feature,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud)
{
	// Get keypoints
	PointCloud<PointXYZ>::Ptr keypointsReference (new PointCloud<PointXYZ>()) ;
	keypointsReference = CalculateKeypoints(filteredReference);
	PointCloud<PointXYZ>::Ptr keypointsCloud (new PointCloud<PointXYZ>()) ;
	keypointsCloud = CalculateKeypoints(filteredCloud);
	cout <<  "Keypoints Calculated" << endl;
	cout << "No of ISS reference keypoints: " << keypointsReference->size() << endl;
	cout << "No of ISS data keypoint: " << keypointsCloud->size() << endl;
	
	// Calculate feature histograms
	FPFHEstimationOMP<PointXYZ, Normal, FPFHSignature33> featureHistogram;
	featureHistogram.setNumberOfThreads(4);

	// Get features for the reference
	PointCloud<FPFHSignature33>::Ptr descriptorsReference (new PointCloud<FPFHSignature33> ());
	search::KdTree<PointXYZ>::Ptr referenceTree (new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud (keypointsReference);
	featureHistogram.setInputNormals (normalReference);
	featureHistogram.setSearchMethod (referenceTree);
	featureHistogram.setRadiusSearch (feature);
	featureHistogram.setSearchSurface(filteredReference);
	featureHistogram.compute(*descriptorsReference);

	// Get features for the cloud to align
	PointCloud<FPFHSignature33>::Ptr descriptorsCloud (new PointCloud<FPFHSignature33> ());
	search::KdTree<PointXYZ>::Ptr cloudTree (new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud (keypointsCloud);
	featureHistogram.setInputNormals (normalCloud);
	featureHistogram.setSearchMethod (cloudTree);
	featureHistogram.setSearchSurface(filteredCloud);
	featureHistogram.compute(*descriptorsCloud);
	cout <<  "Features Calculated" << endl;


	// Deterimine Correspondences
	boost::shared_ptr<Correspondences> correspondences (new Correspondences);
	boost::shared_ptr<Correspondences> remainingCorrespondences (new Correspondences);
	registration::CorrespondenceEstimation<FPFHSignature33,FPFHSignature33> correspondenceEstimation;
	correspondenceEstimation.setInputSource (descriptorsCloud);
	correspondenceEstimation.setInputTarget (descriptorsReference);
	correspondenceEstimation.determineCorrespondences (*correspondences);
	cout << correspondences->size() << " correspondences found." << endl;


	cout << "Rejecting outliers" << endl;
	registration::CorrespondenceRejectorSampleConsensus<PointXYZ> rejector;
	rejector.setInputCloud (keypointsCloud);
	rejector.setInputTarget (keypointsReference);
	rejector.setInputCorrespondences (correspondences);
	rejector.setMaxIterations (400000);
	rejector.setInlierThreshold (inlier);
	rejector.setRefineModel(true);
	rejector.setRefineModel(true);
	rejector.getRemainingCorrespondences(*correspondences, *remainingCorrespondences);
	cout << remainingCorrespondences->size() << " correspondences remaining." << endl;
	cout << "Rejection Complete" << endl;

	if  (correspondences->size()==0)
	{
	rejector.setInlierThreshold (inlier);
	rejector.getCorrespondences (*correspondences);
	cout << "Rejection 2 Complete" << endl;
	}
	
	// get the best transformation and apply it
	Eigen::Matrix4f transformation;
	transformation = rejector.getBestTransformation();
	transformPointCloud (*dataCloud, *dataCloud, transformation);

	printf ("\n");
	console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
	console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
	console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
	console::print_info ("\n");
	console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
	console::print_info ("\n");
}

PointCloud<PointXYZ>::Ptr CopyCloud(PointCloud<PointXYZ>::Ptr input){
	PointCloud<PointXYZ>::Ptr output (new PointCloud<PointXYZ>);
	for (int i = 0; i < input->points.size(); i++){
		PointXYZ point;
		point.x = input->points[i].x;
		point.y = input->points[i].y;
		point.z = input->points[i].z;
		output->points.push_back(point);
	}
	return output;
}

void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud)
{	
	PointCloud<PointXYZ>::Ptr simpleReference = CopyCloud(filteredReference);
	PointCloud<PointXYZ>::Ptr simpleData = CopyCloud(filteredCloud);
	
	// Add the normals to the cloud
	PointCloud<PointNormal>::Ptr referenceWithNormals (new PointCloud<PointNormal>);
	PointCloud<PointNormal>::Ptr cloudWithNormals (new PointCloud<PointNormal>);
	cout << "Concat normals to the filtered reference cloud" << endl;
	concatenateFields(*simpleReference, *normalReference, *referenceWithNormals);
	cout << "Concat normals to the filtered data cloud" << endl;
	concatenateFields(*simpleData, *normalCloud, *cloudWithNormals);
	// output clouds
	PointCloud<PointNormal>::Ptr outputWithNormals (new PointCloud<PointNormal>);
	*outputWithNormals = *cloudWithNormals;
	PointCloud<PointXYZ>::Ptr output (new PointCloud<PointXYZ>);
	*output = *simpleData;


	// Get correspondences
	Eigen::Matrix4f transformFinal (Eigen::Matrix4f::Identity ());
	Eigen::Matrix4f transform_res_from_LM;
	boost::shared_ptr<Correspondences> correspondences (new Correspondences);
	boost::shared_ptr<Correspondences> filteredCorrespondences (new Correspondences);

	int iteration;
	registration::DefaultConvergenceCriteria<float> conv_crit (iteration, transform_res_from_LM, *filteredCorrespondences);
	iteration = 0;
	do
	{
		iteration += 1;
		cout << "Current iteration: " << iteration << endl;

		//compute correspondences as points in the target cloud which have minimum distance to normals computed on the input cloud
		registration::CorrespondenceEstimationNormalShooting<PointNormal, PointNormal, PointNormal> correspondenceEstimation;
		correspondenceEstimation.setInputSource(cloudWithNormals);
		correspondenceEstimation.setInputTarget(referenceWithNormals);
		correspondenceEstimation.setSourceNormals(outputWithNormals);
		correspondenceEstimation.determineCorrespondences(*correspondences);

		// reject based on the angle between the normals at correspondent points
		registration::CorrespondenceRejectorSurfaceNormal::Ptr rej_normals (new registration::CorrespondenceRejectorSurfaceNormal);
		rej_normals->setInputCorrespondences(correspondences);
		double degree = 40;
		double threshold=acos (deg2rad (degree));
		rej_normals->setThreshold (threshold);
		rej_normals->initializeDataContainer<PointNormal, PointNormal> ();
		rej_normals->setInputSource<PointNormal> (outputWithNormals);
		rej_normals->setInputNormals<PointNormal, PointNormal> (cloudWithNormals);
		rej_normals->setInputTarget<PointNormal> (referenceWithNormals);
		rej_normals->setTargetNormals<PointNormal, PointNormal> (referenceWithNormals);
		rej_normals->getCorrespondences (*filteredCorrespondences);

		registration::CorrespondenceRejectorMedianDistance::Ptr rejector2 (new registration::CorrespondenceRejectorMedianDistance);
		rejector2->setInputCorrespondences (filteredCorrespondences);
		rejector2->setMedianFactor(1.3);
		rejector2->getCorrespondences (*filteredCorrespondences);

		registration::TransformationEstimationPointToPlaneLLS<PointNormal, PointNormal,float> trans_est_lm;
		trans_est_lm.estimateRigidTransformation (*outputWithNormals, *referenceWithNormals, *filteredCorrespondences, transform_res_from_LM);
		transformFinal = transformFinal*transform_res_from_LM;

		transformPointCloud (*simpleData, *output, transformFinal);
		transformPointCloudWithNormals(*cloudWithNormals, *outputWithNormals, transformFinal);
		transformPointCloud(*dataCloud, *dataCloud, transformFinal);
	}
	while (!conv_crit.hasConverged ());
}


PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz)
{
	cout << "Original cloud size " << cloud->size() << endl;
	VoxelGrid<PointXYZ> sor3;
	sor3.setInputCloud(cloud);
	sor3.setLeafSize (lx,ly,lz);
	sor3.filter (*filtered);
	cout << "Filtered cloud size " << filtered->size() << endl;
	return filtered;
 }
