#include <cmath>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <math.h>
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

// Segmentation
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>

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

// Logging
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace keywords = boost::log::keywords;
using namespace std;
using namespace pcl;
using namespace boost;

property_tree::ptree CONFIG;
double CLOUD_RESOLUTION;
double REFERENCE_RESOLUTION;

namespace pcl{
struct PointXYZRGBI
{
	PCL_ADD_POINT4D;
	float r;
	float g;
	float b;
	float i;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
}
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBI,           // here we assume a XYZ + "test" (as fields)
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
void FeatureAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud,
	float inlier, float feature,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud);
void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud);
bool LoadCloud(const string &filename, PointCloud<PointXYZRGBI> &cloud, PointCloud<PointXYZ> &simpleCloud);
void Segment(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud);
PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz);
void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud);



// Inline functions
float deg2rad(float deg) {
	float rad = (deg * 3.14159265359) / 180;
	return rad;
}


property_tree::ptree LoadConfig(string configFile) {
	property_tree::ptree pt;
	property_tree::read_json(configFile, pt);
	return pt;
}

void Log(string message) {
	bool log = CONFIG.get("logging.log", false);
	string fname = CONFIG.get("logging.log_file", "");
	bool print = CONFIG.get("logging.print_log", true);
	time_t rawtime;
	struct tm * timeinfo;
	char start[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(start, 80, "%Y_%m_%d %H:%M:%S", timeinfo);
	string output = CONFIG.get("align_file_prefix", "") + ": " + message + "\n";
	if (log) {
		ofstream file(fname);
		if (!file || print)
		{
			cout << output << endl;
			return;
		}
		file << output;
		file.close();
	}
	else {
		cout << output << endl;
	}
	return;
}

void SetupLogger() {
	bool log = CONFIG.get("logging.log", false);
	bool autoGenerate = CONFIG.get("logging.autogenerate_log_file", false);
	string logFile = CONFIG.get("logging.log_file", "");
	string fname = logFile;
	if (log) {
		time_t rawtime;
		struct tm * timeinfo;
		char startstr[80];
		char indexTime[80];

		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(startstr, 80, "_%Y%m%d", timeinfo);
		strftime(indexTime, 80, "%m/%d/%Y %H:%M:%S", timeinfo);
		
		if (autoGenerate) {
			string alignFilePrefix = CONFIG.get("align_file_prefix", "");
			fname = alignFilePrefix + startstr + ".log";
		}
		CONFIG.put("logging.log_file", fname);
		if (ifstream(fname))
		{
			cout << "Logfile already exists logs will be appended." << endl;
		}
		ofstream file(fname);
		if (!file)
		{
			cout << "File could not be created. Logging to console." << endl;
			return;
		}
		file << "Logging start: " << indexTime;
		file.close();
	}
	return;
}

string ValidateConfig() {
	string isValidMessage = "";
	// Check for input files
	string baseFile = CONFIG.get("base_file", "");
	if (baseFile == "") {
		isValidMessage += "\tbase_file: A base file for alignment is required.\n";
	}
	string snipFile = CONFIG.get("snip_file", "");
	string alignFilePrefix = CONFIG.get("align_file_prefix", "");
	if (alignFilePrefix == "") {
		isValidMessage += "\talign_file_prefix: The prefix for the files to align is required.\n";
	}
	int numFiles = CONFIG.get("number_files", 0);
	if (numFiles <= 0) {
		isValidMessage += "\tnumber_files: Number of files must be greater than zero.\n";
	}
	string fileExtension = CONFIG.get("file_extension", "");
	if (fileExtension == "") {
		isValidMessage += "\tfile_extension: A file extension for the alignment files is required.\n";
	}

	// Check for logging
	bool log = CONFIG.get("logging.log", false);
	if (log) {
		bool autoGenerate = CONFIG.get("logging.autogenerate_log_file", false);
		string logFile = CONFIG.get("logging.log_file", "");
		if (!autoGenerate && logFile == "") {
			isValidMessage += "\tlogging.log_file: Without specifying autogeneration of log files, a file name is required.\n";
		}
	}

	return isValidMessage;
}


int main(int argc, char** argv) {
	string configFile = argv[1];
	CONFIG = LoadConfig(configFile);
	string isValidMessage = ValidateConfig();

	if (isValidMessage != "") {
		cout << "Invalid config parameters: " + isValidMessage << endl;
		return (-1);
	}
	SetupLogger();


	// Load reference cloud
	PointCloud<PointXYZRGBI>::Ptr cloud1(new PointCloud<PointXYZRGBI>);
	PointCloud<PointXYZ>::Ptr simpleCloud1(new PointCloud<PointXYZ>);
	string baselineFile = CONFIG.get("base_file", "");
	string alignFilePrefix = CONFIG.get("align_file_prefix", "");
	string fileExtension = CONFIG.get("file_extension", "");
	if (fileExtension.at(0) != '.') {
		fileExtension = "." + fileExtension;
	}
	if (!LoadCloud(baselineFile, *cloud1, *simpleCloud1)) {
		Log("Unable to load file: " + baselineFile);
		return (-1);
	}


	PointCloud<PointXYZRGBI>::Ptr mergedCloud(new PointCloud<PointXYZRGBI>);
	int numFiles = CONFIG.get("number_files", 0);
	for (int i = 1; i <= numFiles; i++) {

		// Loading cloud to align
		PointCloud<PointXYZ>::Ptr simpleCloud2(new PointCloud<PointXYZ>);
		PointCloud<PointXYZRGBI>::Ptr cloud2(new PointCloud<PointXYZRGBI>);
		string cloudFile = alignFilePrefix + to_string((long long)i) + fileExtension;
		if (!LoadCloud(cloudFile, *cloud2, *simpleCloud2)) {
			Log("Unable to load file: " + cloudFile);
			return (-1);
		}
		Log("Clouds successfully loaded.");
		Segment(cloud2, simpleCloud2);

		string ftext = alignFilePrefix + "_" + to_string((long long)i) + "_aligned.txt";

		float leafSize = CONFIG.get("feature_align.voxel_leafsize", 0.1);
		float normalRadius = CONFIG.get("feature_align.normal_radius", 0.8);
		float featureSize = CONFIG.get("feature_align.feature_size", 1.6);
		float inlier = CONFIG.get("feature_align.voxel_leafsize", .25);
		if (CONFIG.get("feature_align.use_resolution", false) || CONFIG.get("icp_align.use_resolution", false)) {
			CLOUD_RESOLUTION = CalculateResolution(simpleCloud2);
			leafSize *= CLOUD_RESOLUTION;
			normalRadius *= CLOUD_RESOLUTION;
			featureSize *= CLOUD_RESOLUTION;
		}
		PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
		PointCloud<PointXYZ>::Ptr filteredReference(new PointCloud<PointXYZ>);
		PointCloud<Normal>::Ptr normalReference(new PointCloud<Normal>);
		PointCloud<Normal>::Ptr normalCloud(new PointCloud<Normal>);
		for (int i = 0; i < 3; i++) {
			*filteredCloud = *simpleCloud2;
			*filteredReference = *simpleCloud1;
			leafSize -= leafSize * (float)i / 20;
			normalRadius += normalRadius * (float)i / 20;
			inlier -= 0.01;
			featureSize -= featureSize * (float)i / 20;

			// Perform the voxel filtering
			filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, leafSize, leafSize, leafSize);
			filteredReference = VoxelFilter(simpleCloud1, filteredReference, leafSize, leafSize, leafSize);

			// Calculate normals
			Log("Calculating Normals with radius: " + to_string((long long)normalRadius));
			normalReference = CalculateNormals(filteredReference, normalRadius);
			normalCloud = CalculateNormals(filteredCloud, normalRadius);

			Log("Starting feature-based alignment with inlier and feature size: " + to_string((long long)inlier) + " and " + to_string((long long)featureSize));
			FeatureAlign(cloud2, simpleCloud2, inlier, featureSize, filteredReference,
				filteredCloud, normalReference, normalCloud);
		}

		float icpLeafsize = CONFIG.get("icp_align.voxel_leafsize", 8);
		float icpNormals = CONFIG.get("icp_align.normal_radius", 40);
		if (CONFIG.get("icp_align.use_resolution", false)) {
			icpLeafsize *= CLOUD_RESOLUTION;
			icpNormals *= CLOUD_RESOLUTION;
		}

		*filteredCloud = *simpleCloud2;
		*filteredReference = *simpleCloud1;

		filteredReference = VoxelFilter(simpleCloud1, filteredReference, icpLeafsize, icpLeafsize, icpLeafsize);
		filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, icpLeafsize, icpLeafsize, icpLeafsize);
		normalReference = CalculateNormals(filteredReference, icpNormals);
		normalCloud = CalculateNormals(filteredCloud, icpNormals);
		IterativeAlign(cloud2, simpleCloud2, filteredReference, filteredCloud, normalReference, normalCloud);

		if (CONFIG.get("icp_align.repetitions.repeat", false)) {
			int repetitions = CONFIG.get("icp_align.repetitions.number_repetitions", 3);
			float change = CONFIG.get("icp_align.repetitions.voxel_change", -0.1);
			for (int icp = 0; icp < repetitions; icp++) {
				Log("Repeating ICP");
				icpLeafsize += change* icpLeafsize;
				icpNormals += change * icpNormals;
				if (icpLeafsize < 0 || icpNormals < 0) {
					continue;
				}

				*filteredCloud = *simpleCloud2;
				*filteredReference = *simpleCloud1;

				filteredReference = VoxelFilter(simpleCloud1, filteredReference, icpLeafsize, icpLeafsize, icpLeafsize);
				filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, icpLeafsize, icpLeafsize, icpLeafsize);
				if ((int)filteredReference->size() < 1000 || (int)filteredCloud->size() < 1000) {
					continue;
				}
				normalReference = CalculateNormals(filteredReference, icpNormals);
				normalCloud = CalculateNormals(filteredCloud, icpNormals);
				IterativeAlign(cloud2, simpleCloud2, filteredReference, filteredCloud, normalReference, normalCloud);
			}
		}
		WriteCloud(ftext, cloud2);
		if (i == 1) {
			mergedCloud->points = cloud2->points;
		}
		else {
			mergedCloud->points.insert(mergedCloud->points.end(), cloud2->points.begin(), cloud2->points.end());
		}
	}
	WriteCloud(alignFilePrefix + "_merged.txt", mergedCloud);

	return (0);
}

PointCloud<PointXYZ>::Ptr CalculateKeypoints(PointCloud<PointXYZ>::Ptr &cloud)
{
	PointCloud<PointXYZ>::Ptr keypoints(new PointCloud<PointXYZ>);
	search::KdTree<PointXYZ>::Ptr tree = search::KdTree<PointXYZ>::Ptr(new search::KdTree<PointXYZ>);
	*keypoints = *cloud;

	ISSKeypoint3D<PointXYZ, PointXYZ> keypointDetector;
	keypointDetector.setInputCloud(cloud);
	keypointDetector.setSearchMethod(tree);
	double resolution = CalculateResolution(cloud);
	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	keypointDetector.setSalientRadius(6 * resolution);
	//cout << resolution << endl;
	// Set the radius for the application of the non maxima supression algorithm.
	keypointDetector.setNonMaxRadius(4 * resolution);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	keypointDetector.setNormalRadius(4 * resolution);
	keypointDetector.setBorderRadius(resolution);
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
	normalEstimation.setViewPoint(0, 0, 0);
	normalEstimation.setInputCloud(cloud);

	// An empty kdtree is required for searching
	search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
	normalEstimation.setSearchMethod(tree);
	// Output datasets
	PointCloud<Normal>::Ptr cloudNormals(new PointCloud<Normal>);
	// Use all neighbors in a sphere of radius RAD
	normalEstimation.setRadiusSearch(radius);
	normalEstimation.setNumberOfThreads(8);
	normalEstimation.compute(*cloudNormals);
	return cloudNormals;
}

double CalculateResolution(const PointCloud<PointXYZ>::ConstPtr& cloud) {
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
		if (!pcl_isfinite((*cloud)[i].x))
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

	Log("Average resolution: " + to_string((long long)resolution));

	return resolution;
}

double CalculateMeanDistance(const PointCloud<PointXYZ>::ConstPtr& cloud) {
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
		if (!pcl_isfinite((*cloud)[i].x))
			continue;

		// Consider the nearest neighbor not including the point itself
		numberOfNeighbors = tree.nearestKSearch(i, 8, indices, squaredDistances);
		if (numberOfNeighbors == 8)
		{
			for (int i = 1; i < 8; i++) {
				distance += squaredDistances[i];
				++numberOfPoints;
			}
		}
	}
	if (numberOfPoints != 0)
		distance /= numberOfPoints;

	return distance;
}

void FeatureAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud,
	float inlier, float feature,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud)
{
	// Get keypoints
	// Get keypoints
	for (int i = 0; i < (int)normalReference->size(); i++) {
		if (to_string(normalReference->points[i].normal_x) == "nan" || to_string(normalReference->points[i].normal_y) == "nan" || to_string(normalReference->points[i].normal_z) == "nan") {
			normalReference->points[i].normal_x = 0;
			normalReference->points[i].normal_y = 0;
			normalReference->points[i].normal_z = 0;
		}
	}

	PointCloud<PointXYZ>::Ptr keypointsReference(new PointCloud<PointXYZ>());
	keypointsReference = CalculateKeypoints(filteredReference);
	PointCloud<PointXYZ>::Ptr keypointsCloud(new PointCloud<PointXYZ>());
	keypointsCloud = CalculateKeypoints(filteredCloud);
	Log( "Keypoints Calculated");
	Log("No of ISS reference keypoints: " + to_string((long long)keypointsReference->size()));
	Log("No of ISS data keypoint: " + to_string((long long)keypointsCloud->size()));

	// Calculate feature histograms
	FPFHEstimationOMP<PointXYZ, Normal, FPFHSignature33> featureHistogram;
	featureHistogram.setNumberOfThreads(4);
	// Get features for the reference
	PointCloud<FPFHSignature33>::Ptr descriptorsReference(new PointCloud<FPFHSignature33>());
	search::KdTree<PointXYZ>::Ptr referenceTree(new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud(keypointsReference);
	featureHistogram.setInputNormals(normalReference);
	featureHistogram.setSearchMethod(referenceTree);
	featureHistogram.setRadiusSearch(feature);
	featureHistogram.setSearchSurface(filteredReference);
	featureHistogram.compute(*descriptorsReference);

	// Get features for the cloud to align
	PointCloud<FPFHSignature33>::Ptr descriptorsCloud(new PointCloud<FPFHSignature33>());
	search::KdTree<PointXYZ>::Ptr cloudTree(new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud(keypointsCloud);
	featureHistogram.setInputNormals(normalCloud);
	featureHistogram.setSearchMethod(cloudTree);
	featureHistogram.setSearchSurface(filteredCloud);
	featureHistogram.compute(*descriptorsCloud);

	// Deterimine Correspondences
	boost::shared_ptr<Correspondences> correspondences(new Correspondences);
	boost::shared_ptr<Correspondences> remainingCorrespondences(new Correspondences);
	registration::CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> correspondenceEstimation;
	correspondenceEstimation.setInputSource(descriptorsCloud);
	correspondenceEstimation.setInputTarget(descriptorsReference);
	correspondenceEstimation.determineCorrespondences(*correspondences);
	Log(to_string((long long)correspondences->size()) + " correspondences found");

	Log("Rejecting outliers");
	registration::CorrespondenceRejectorSampleConsensus<PointXYZ> rejector;
	rejector.setInputSource(keypointsCloud);
	rejector.setInputTarget(keypointsReference);
	rejector.setInputCorrespondences(correspondences);
	rejector.setMaximumIterations(400000);
	rejector.setInlierThreshold(inlier);
	rejector.setRefineModel(true);
	rejector.getRemainingCorrespondences(*correspondences, *remainingCorrespondences);
	Log(to_string((long long)remainingCorrespondences->size()) + " correspondences remaining");
	Log("Rejection Complete");

	if (correspondences->size() == 0)
	{
		rejector.setInlierThreshold(inlier);
		rejector.getCorrespondences(*correspondences);
		Log("Rejection 2 Complete");
	}

	// get the best transformation and apply it
	Eigen::Matrix4f transformation;
	transformation = rejector.getBestTransformation();
	transformPointCloud(*simpleCloud, *simpleCloud, transformation);
	transformPointCloud(*dataCloud, *dataCloud, transformation);

	ostringstream stringStream;
	stringStream << "R = " << endl;
	stringStream << "\t\t|\t" << transformation(0, 0) << "\t " << transformation(0, 1) << "\t " << transformation(0, 2) << "\t|" << endl;
	stringStream << "\t\t|\t" << transformation(1, 0) << "\t " << transformation(1, 1) << "\t " << transformation(1, 2) << "\t|" << endl;
	stringStream << "\t\t|\t" << transformation(2, 0) << "\t " << transformation(2, 1) << "\t " << transformation(2, 2) << "\t|" << endl;
	stringStream << "\t\t t = " << transformation(0, 3) << ", " << transformation(1, 3) << ", " << transformation(2, 3) << " >" << endl;
	Log(stringStream.str());
}

PointCloud<PointXYZ>::Ptr CopyCloud(PointCloud<PointXYZ>::Ptr input) {
	PointCloud<PointXYZ>::Ptr output(new PointCloud<PointXYZ>);
	for (int i = 0; i < input->points.size(); i++) {
		PointXYZ point;
		point.x = input->points[i].x;
		point.y = input->points[i].y;
		point.z = input->points[i].z;
		output->points.push_back(point);
	}
	return output;
}

void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud)
{

	// Add the normals to the cloud
	PointCloud<PointNormal>::Ptr referenceWithNormals(new PointCloud<PointNormal>);
	PointCloud<PointNormal>::Ptr cloudWithNormals(new PointCloud<PointNormal>);
	Log("Concat normals to the filtered reference cloud");
	concatenateFields(*filteredReference, *normalReference, *referenceWithNormals);
	Log("Concat normals to the filtered data cloud");
	concatenateFields(*filteredCloud, *normalCloud, *cloudWithNormals);
	// output clouds
	PointCloud<PointNormal>::Ptr outputWithNormals(new PointCloud<PointNormal>);
	*outputWithNormals = *cloudWithNormals;
	PointCloud<PointXYZ>::Ptr output(new PointCloud<PointXYZ>);
	*output = *filteredCloud;


	// Get correspondences
	Eigen::Matrix4f transformFinal(Eigen::Matrix4f::Identity());
	Eigen::Matrix4f transform_res_from_LM;
	boost::shared_ptr<Correspondences> correspondences(new Correspondences);
	boost::shared_ptr<Correspondences> filteredCorrespondences(new Correspondences);

	int iteration;
	registration::DefaultConvergenceCriteria<float> conv_crit(iteration, transform_res_from_LM, *filteredCorrespondences);
	iteration = 0;
	do
	{
		iteration += 1;
		Log("Current iteration: " + to_string((long long)iteration));

		//compute correspondences as points in the target cloud which have minimum distance to normals computed on the input cloud
		registration::CorrespondenceEstimationNormalShooting<PointNormal, PointNormal, PointNormal> correspondenceEstimation;
		correspondenceEstimation.setInputSource(cloudWithNormals);
		correspondenceEstimation.setInputTarget(referenceWithNormals);
		correspondenceEstimation.setSourceNormals(outputWithNormals);
		correspondenceEstimation.determineCorrespondences(*correspondences);

		// reject based on the angle between the normals at correspondent points
		registration::CorrespondenceRejectorSurfaceNormal::Ptr rej_normals(new registration::CorrespondenceRejectorSurfaceNormal);
		rej_normals->setInputCorrespondences(correspondences);
		double degree = 40;
		double threshold = acos(deg2rad(degree));
		rej_normals->setThreshold(threshold);
		rej_normals->initializeDataContainer<PointNormal, PointNormal>();
		rej_normals->setInputSource<PointNormal>(outputWithNormals);
		rej_normals->setInputNormals<PointNormal, PointNormal>(cloudWithNormals);
		rej_normals->setInputTarget<PointNormal>(referenceWithNormals);
		rej_normals->setTargetNormals<PointNormal, PointNormal>(referenceWithNormals);
		rej_normals->getCorrespondences(*filteredCorrespondences);

		registration::CorrespondenceRejectorMedianDistance::Ptr rejector2(new registration::CorrespondenceRejectorMedianDistance);
		rejector2->setInputCorrespondences(filteredCorrespondences);
		rejector2->setMedianFactor(1.3);
		rejector2->getCorrespondences(*filteredCorrespondences);

		registration::TransformationEstimationPointToPlaneLLS<PointNormal, PointNormal, float> trans_est_lm;
		trans_est_lm.estimateRigidTransformation(*outputWithNormals, *referenceWithNormals, *filteredCorrespondences, transform_res_from_LM);
		transformFinal = transformFinal * transform_res_from_LM;

		transformPointCloud(*filteredCloud, *output, transformFinal);
		transformPointCloudWithNormals(*cloudWithNormals, *outputWithNormals, transformFinal);
		transformPointCloud(*dataCloud, *dataCloud, transformFinal);
		transformPointCloud(*simpleCloud, *simpleCloud, transformFinal);
	} while (!conv_crit.hasConverged());

}

bool LoadCloud(const string &filename, PointCloud<PointXYZRGBI> &cloud, PointCloud<PointXYZ> &simpleCloud)
{
	ifstream fs;
	fs.open(filename.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail())
	{
		Log("Could not open file: " + filename);
		fs.close();
		return (false);
	}

	string line;
	vector<string> st;

	while (!fs.eof())
	{
		getline(fs, line);
		// Ignore empty lines
		if (line == "" || line.at(0) == '/')
			continue;

		// Tokenize the line
		boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r, "), boost::token_compress_on);
		float r, g, b, i;
		if (st.size() < 3) {
			continue;
		}
		else if (st.size() == 3) {
			r = 0;
			g = 0;
			b = 0;
			i = 0;
		}
		else if (st.size() == 4) {
			r = 0;
			g = 0;
			b = 0;
			i = float(atof(st[3].c_str()));
		}
		else if (st.size() == 6) {
			r = float(atof(st[3].c_str()));
			g = float(atof(st[4].c_str()));
			b = float(atof(st[5].c_str()));
			i = 0;
		}
		else if (st.size() >= 7) {
			r = float(atof(st[4].c_str()));
			g = float(atof(st[5].c_str()));
			b = float(atof(st[6].c_str()));
			i = float(atof(st[3].c_str()));
		}
		PointXYZRGBI point;
		point.x = float(atof(st[0].c_str()));
		point.y = float(atof(st[1].c_str()));
		point.z = float(atof(st[2].c_str()));
		point.r = r;
		point.g = g;
		point.b = b;
		point.i = i;
		cloud.push_back(point);

		PointXYZ simplePoint;
		simplePoint.x = float(atof(st[0].c_str()));
		simplePoint.y = float(atof(st[1].c_str()));
		simplePoint.z = float(atof(st[2].c_str()));
		simpleCloud.push_back(simplePoint);
	}
	fs.close();

	cloud.width = uint32_t(cloud.size()); cloud.height = 1; cloud.is_dense = true;
	return (true);
}

void Segment(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud) {
	string polylineFile = CONFIG.get("segmentation_file", "");
	if (polylineFile != "") {
		Log("Segmentation file found: " + polylineFile + ". Attempting to crop.");
		PointCloud<PointXYZRGBI>::Ptr polyline(new PointCloud<PointXYZRGBI>);
		PointCloud<PointXYZ>::Ptr simplePolyline(new PointCloud<PointXYZ>);
		if (!LoadCloud(polylineFile, *polyline, *simplePolyline)) {
			Log("Unable to load segmentation file: " + polylineFile);
			return;
		}

		PointCloud<PointXYZ>::Ptr hullPoints(new PointCloud<PointXYZ>);
		vector<Vertices> hullPolygons;
		ConvexHull<PointXYZ> cHull;
		cHull.setInputCloud(simplePolyline);
		cHull.setDimension(3);
		cHull.reconstruct(*hullPoints, hullPolygons);

		CropHull<PointXYZ> cropHullFilter;
		cropHullFilter.setHullIndices(hullPolygons);
		cropHullFilter.setHullCloud(hullPoints);
		cropHullFilter.setDim(3);
		cropHullFilter.setCropOutside(true);
		cropHullFilter.setInputCloud(simpleCloud);
		vector<int> indices;
		cropHullFilter.filter(indices);

		PointCloud<PointXYZRGBI>::Ptr filtered(new PointCloud<PointXYZRGBI>);
		filtered->width = indices.size();
		filtered->height = 1;
		filtered->is_dense = false;
		filtered->points.resize(filtered->width * filtered->height);
		PointCloud<PointXYZ>::Ptr simpleFiltered(new PointCloud<PointXYZ>);
		simpleFiltered->width = indices.size();
		simpleFiltered->height = 1;
		simpleFiltered->is_dense = false;
		simpleFiltered->points.resize(simpleFiltered->width * simpleFiltered->height);

		for (size_t i = 0; i < indices.size(); i++) {
			int idx = indices[i];
			PointXYZ simple = simpleCloud->points[idx];
			PointXYZRGBI point = cloud->points[idx];
			filtered->points[i] = point;
			simpleFiltered->points[i] = simple;
		}
		*cloud = *filtered;
		*simpleCloud = *simpleFiltered;
	}
	return;
}

PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz)
{
	Log("Original cloud size " + to_string((long long)cloud->size()));
	VoxelGrid<PointXYZ> sor3;
	sor3.setInputCloud(cloud);
	sor3.setLeafSize(lx, ly, lz);
	sor3.filter(*filtered);
	Log("Filtered cloud size " + to_string((long long)cloud->size()));;
	return filtered;
}

void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud) {
	ofstream file(filename);
	if (!file)
	{
		Log("File could not be created.");
		return;
	}
	file << "\\X Y Z R G B I" << endl;
	for (size_t i = 0; i < cloud->points.size(); i++) {
		float x = cloud->points[i].x;
		float y = cloud->points[i].y;
		float z = cloud->points[i].z;
		float r = cloud->points[i].r;
		float g = cloud->points[i].g;
		float b = cloud->points[i].b;
		float intensity = cloud->points[i].i;
		string tab = "\t";
		file << x << tab << y << tab << z << tab << r << tab << g << tab << b << tab << intensity << endl;
	}
	file.close();
	Log("Cloud written to " + filename);
	return;
}
