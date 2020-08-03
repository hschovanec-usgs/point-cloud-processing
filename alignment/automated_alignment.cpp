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
#include <pcl/filters/uniform_sampling.h>

// PCL IO
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

// Normals
#include <pcl/features/normal_3d_omp.h>


// Boost property tree
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// Segmentation and Filtering
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>

#include <pcl/visualization/pcl_visualizer.h>

// Registration
#include <pcl/features/fpfh_omp.h>
#include <pcl/Keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_svd.h>

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
using namespace Eigen;

property_tree::ptree CONFIG;
double CLOUD_RESOLUTION;
double REFERENCE_RESOLUTION;

namespace pcl {
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
void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud, PointCloud<PointXYZ>::Ptr &simpleReference,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud, float voxel, float normal, float median, float degree);
bool LoadCloud(const string &filename, PointCloud<PointXYZRGBI> &cloud, PointCloud<PointXYZ> &simpleCloud);
void Segment(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud, string polylineFile);
void UniformFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float radius);
void UniformFilter(const PointCloud<PointXYZRGB>::Ptr cloud, PointCloud<PointXYZRGB>::Ptr &filtered, float radius);
PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz);
void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud);
void SORFilter(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud);



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
	bool log = CONFIG.get<bool>("logging.log", false);
	string fname = CONFIG.get<string>("logging.log_file", "");
	bool print = CONFIG.get<bool>("logging.print_log", true);
	time_t rawtime;
	struct tm * timeinfo;
	char start[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(start, 80, "%Y_%m_%d %H:%M:%S", timeinfo);
	string output = start;
	output += ": " + message + "\n";
	if (log) {
		fstream file;
		file.open(fname, fstream::app);
		if (!file && print)
		{
			cout << output;
			return;
		}
		else if (print) {
			cout << output;
		}
		file << output;
		file.close();
	}
	else {
		cout << output;
	}
	return;
}

void SetupLogger() {
	bool log = CONFIG.get<bool>("logging.log", false);
	bool autoGenerate = CONFIG.get<bool>("logging.autogenerate_log_file", false);
	string logFile = CONFIG.get<string>("logging.log_file", "");
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
			string alignFilePrefix = CONFIG.get<string>("align_file_prefix", "");
			string writeFilePrefix = "";
			int loc = alignFilePrefix.find_last_of("\\");
			if (loc == string::npos) {
				writeFilePrefix = alignFilePrefix;
			}
			else {
				writeFilePrefix = alignFilePrefix.substr(loc + 1);
			}
			fname = alignFilePrefix + startstr + ".log";
		}
		CONFIG.put("logging.log_file", fname);
		cout << "Logging to " << fname << endl;
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
		file << "-------------------- Logging start: " << indexTime << " --------------------" << endl;
		file.close();
	}
	return;
}

string ValidateConfig() {
	string isValidMessage = "";
	// Check for input files
	string baseFile = CONFIG.get<string>("base_file", "");
	if (baseFile == "") {
		isValidMessage += "\tbase_file: A base file for alignment is required.\n";
	}
	string snipFile = CONFIG.get<string>("snip_file", "");
	string alignFilePrefix = CONFIG.get<string>("align_file_prefix", "");
	if (alignFilePrefix == "") {
		isValidMessage += "\talign_file_prefix: The prefix for the files to align is required.\n";
	}
	int numFiles = CONFIG.get<int>("number_files", 0);
	if (numFiles <= 0) {
		isValidMessage += "\tnumber_files: Number of files must be greater than zero.\n";
	}
	string fileExtension = CONFIG.get<string>("file_extension", "");
	if (fileExtension == "") {
		isValidMessage += "\tfile_extension: A file extension for the alignment files is required.\n";
	}

	// Check for logging
	bool log = CONFIG.get<bool>("logging.log", false);
	if (log) {
		bool autoGenerate = CONFIG.get<bool>("logging.autogenerate_log_file", false);
		string logFile = CONFIG.get<string>("logging.log_file", "");
		if (!autoGenerate && logFile == "") {
			isValidMessage += "\tlogging.log_file: Without specifying autogeneration of log files, a file name is required.\n";
		}
	}

	return isValidMessage;
}

void TranslateCentroid(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud, vector<double> referenceCentroid, vector<double> dataCentroid) {
	float dx = referenceCentroid[0] - dataCentroid[0];
	float dy = referenceCentroid[1] - dataCentroid[1];
	float dz = referenceCentroid[2] - dataCentroid[2];

	ostringstream transMSG;
	transMSG << "Translating centroid using vector: <" << dx << ", " << dy << ", " << dz << ">";
	Log(transMSG.str());


	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x += dx;
		cloud->points[i].y += dy;
		cloud->points[i].z += dz;
		simpleCloud->points[i].x += dx;
		simpleCloud->points[i].y += dy;
		simpleCloud->points[i].z += dz;
	}
	Log("Translation complete");
	return;
}

vector<double> GetCentroid(PointCloud<PointXYZ>::Ptr cloud) {
	vector<double> centroid(3);
	MatrixXf subpoints(3, cloud->points.size());
	for (size_t s = 0; s < cloud->points.size(); s++) {
		subpoints(0, s) = cloud->points[s].x;
		subpoints(1, s) = cloud->points[s].y;
		subpoints(2, s) = cloud->points[s].z;
	}
	MatrixXf subcentroid = subpoints.rowwise().mean();
	centroid[0] = subcentroid(0);
	centroid[1] = subcentroid(1);
	centroid[2] = subcentroid(2);

	return centroid;
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
	string baselineFile = CONFIG.get<string>("base_file", "");
	string outputFolder = CONFIG.get<string>("output_folder", ".");
	string alignFilePrefix = CONFIG.get<string>("align_file_prefix", "");
	int loc = alignFilePrefix.find_last_of("\\");
	string writeFilePrefix = "";
	if (loc == string::npos) {
		writeFilePrefix = alignFilePrefix;
	}
	else {
		writeFilePrefix = alignFilePrefix.substr(loc + 1);
	}

	string fileExtension = CONFIG.get<string>("file_extension", "");
	if (fileExtension.at(0) != '.') {
		fileExtension = "." + fileExtension;
	}
	Log("Loading base file: " + baselineFile);

	if (!LoadCloud(baselineFile, *cloud1, *simpleCloud1)) {
		Log("Unable to load file: " + baselineFile);
		return (-1);
	}

	vector<double> referenceCentroid = GetCentroid(simpleCloud1);
	ostringstream rcMSG;
	rcMSG << "Reference centroid: <" << referenceCentroid[0] << ", " << referenceCentroid[1] << ", " << referenceCentroid[2] <<  ">";
	Log(rcMSG.str());
	REFERENCE_RESOLUTION = CalculateResolution(simpleCloud1);


	PointCloud<PointXYZRGBI>::Ptr mergedCloud(new PointCloud<PointXYZRGBI>);
	PointCloud<PointXYZ>::Ptr mergedSimple(new PointCloud<PointXYZ>);
	int numFiles = CONFIG.get<int>("number_files", 0);
	for (int i = 1; i <= numFiles; i++) {

		// Loading cloud to align
		PointCloud<PointXYZ>::Ptr simpleCloud2(new PointCloud<PointXYZ>);
		PointCloud<PointXYZRGBI>::Ptr cloud2(new PointCloud<PointXYZRGBI>);
		string cloudFile = alignFilePrefix + to_string((long long)i) + fileExtension;
		Log("Loading data file: " + cloudFile);
		if (!LoadCloud(cloudFile, *cloud2, *simpleCloud2)) {
			Log("Unable to load file: " + cloudFile);
			return (-1);
		}
		Log("Clouds successfully loaded.");

		// Segment cloud
		if (CONFIG.get<bool>("segment.segment", false)) {
			string polylineFile = CONFIG.get<string>("segment.initial_snip_file", "");
			if (polylineFile == "") {
				Log("No snip file provided. Continuing...");
			} {
				Segment(cloud2, simpleCloud2, polylineFile);
			}
		}

		// Performing initial translation
		vector<double> dataCentroid = GetCentroid(simpleCloud2);
		ostringstream dcMSG;
		dcMSG << "Data centroid: <" << dataCentroid[0] << ", " << dataCentroid[1] << ", " << dataCentroid[2] << ">";
		Log(dcMSG.str());
		TranslateCentroid(cloud2, simpleCloud2, referenceCentroid, dataCentroid);

		string ftext = writeFilePrefix + "_" + to_string((long long)i) + "_aligned.txt";

		float leafSize = CONFIG.get<float>("feature_align.voxel_leafsize", 0.1);
		float normalRadius = CONFIG.get<float>("feature_align.normal_radius", 0.8);
		float featureSize = CONFIG.get<float>("feature_align.feature_size", 1.6);
		float inlier_mult = CONFIG.get<float>("feature_align.inlier", 1);
		CLOUD_RESOLUTION = CalculateResolution(simpleCloud2);
		if (CONFIG.get<bool>("feature_align.use_resolution", false)) {

			leafSize *= CLOUD_RESOLUTION;
			normalRadius *= CLOUD_RESOLUTION;
			featureSize *= CLOUD_RESOLUTION;
		}
		PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
		PointCloud<PointXYZ>::Ptr filteredReference(new PointCloud<PointXYZ>);
		PointCloud<Normal>::Ptr normalReference(new PointCloud<Normal>);
		PointCloud<Normal>::Ptr normalCloud(new PointCloud<Normal>);
		int repetitions = 1;
		float change = 0;
		if (CONFIG.get<bool>("feature_align.repetitions.repeat", false)) {
			repetitions = CONFIG.get<int>("feature_align.repetitions.number_repetitions", 3) + 1;
			change = CONFIG.get<float>("feature_align.repetitions.voxel_change", -0.1);
		}
		for (int feat = 0; feat < repetitions; feat++) {
			if (feat != 0) {
				*filteredCloud = *simpleCloud2;
				*filteredReference = *simpleCloud1;
				leafSize += change * leafSize;
				normalRadius += change * normalRadius;
				featureSize += change * featureSize;
			}
			cout << "LEAFSIZE HERE: " << leafSize << endl;
			// Perform the voxel filtering
			filteredReference = VoxelFilter(simpleCloud1, filteredReference, leafSize, leafSize, leafSize);
			filteredCloud = VoxelFilter(simpleCloud2, filteredCloud, leafSize, leafSize, leafSize);
			double inlier = inlier_mult;
			// Calculate normals
			ostringstream normalMSG;
			normalMSG << "Calculating Normals with radius: " << normalRadius;
			Log(normalMSG.str());
			normalReference = CalculateNormals(filteredReference, normalRadius);
			normalCloud = CalculateNormals(filteredCloud, normalRadius);

			ostringstream featureMSG;
			featureMSG << "Starting feature-based alignment with inlier and feature size: " << inlier << " and " << featureSize;
			Log(featureMSG.str());
			FeatureAlign(cloud2, simpleCloud2, inlier, featureSize, filteredReference,
				filteredCloud, normalReference, normalCloud);


				// Segment cloud
				/**if (CONFIG.get<bool>("segment.segment", false)) {
					string polylineFile = CONFIG.get<string>("segment.feature_snip", "");
					if (polylineFile == "") {
						Log("No snip file provided. Continuing...");
					} {

						PointCloud<PointXYZRGBI>::Ptr cloudCopy(new PointCloud<PointXYZRGBI>);
						PointCloud<PointXYZ>::Ptr simpleCopy(new PointCloud<PointXYZ>);
						*cloudCopy = *cloud2;
						*simpleCopy = *simpleCloud2;
						Segment(cloudCopy, simpleCopy, polylineFile);

						// Perform the voxel filtering
						filteredReference = VoxelFilter(simpleCloud1, filteredReference, leafSize, leafSize, leafSize);
						filteredCloud = VoxelFilter(simpleCopy, filteredCloud, leafSize, leafSize, leafSize);

						// Calculate normals
						ostringstream normalMSG;
						normalMSG << "Calculating Normals with radius: " << normalRadius;
						Log(normalMSG.str());
						normalReference = CalculateNormals(filteredReference, normalRadius);
						normalCloud = CalculateNormals(filteredCloud, normalRadius);
						inlier = inlier_mult;
						ostringstream featureMSG;
						featureMSG << "Starting feature-based alignment with inlier and feature size: " << inlier << " and " << featureSize;
						Log(featureMSG.str());
						FeatureAlign(cloud2, simpleCopy, inlier, featureSize, filteredReference,
							filteredCloud, normalReference, normalCloud);
					}
				}**/
				
				
		}

		// Segment cloud
		if (CONFIG.get<bool>("segment.segment", false)) {
			string polylineFile = CONFIG.get<string>("segment.final_snip_file", "");
			if (polylineFile == "") {
				Log("No snip file provided. Continuing...");
			} {
				Segment(cloud2, simpleCloud2, polylineFile);
			}
		}
		// Perform SOR filter
		SORFilter(cloud2, simpleCloud2);

		WriteCloud(outputFolder + "\\" + writeFilePrefix + "_" + to_string((long long)i) + "_roughalign.pts", cloud2);


		float icpLeafsize = CONFIG.get<float>("icp_align.voxel_leafsize", 8);
		float icpNormals = CONFIG.get<float>("icp_align.normal_radius", 40);
		if (CONFIG.get<bool>("icp_align.use_resolution", false)) {
			icpLeafsize *= CLOUD_RESOLUTION;
		}
		repetitions = 1;
		change = 0;
		if (CONFIG.get<bool>("icp_align.repetitions.repeat", false)) {
			repetitions = CONFIG.get<int>("icp_align.repetitions.number_repetitions", 3) + 1;
			change = CONFIG.get<float>("icp_align.repetitions.voxel_change", -0.1);
		}
		cout << "ICP LEAFSIZE: " << icpLeafsize << endl;

		PointCloud<PointXYZRGBI>::Ptr cloudCopy(new PointCloud<PointXYZRGBI>);
		PointCloud<PointXYZ>::Ptr simpleCopy(new PointCloud<PointXYZ>);
		PointCloud<PointXYZRGBI>::Ptr referenceCopy(new PointCloud<PointXYZRGBI>);
		PointCloud<PointXYZ>::Ptr simpleReferenceCopy(new PointCloud<PointXYZ>);

		double degree = CONFIG.get<float>("icp_align.threshold_degree", 40);
		double median = CONFIG.get<float>("icp_align.median_factor", 1.3);
		for (int icp = 0; icp < repetitions; icp++) {
			if (icp != 0) {
				icpLeafsize += change * icpLeafsize;
				icpNormals += change * icpNormals;

				icpNormals += change * icpNormals;
				icpNormals += change * icpNormals;
				//degree += change * degree;
				//median += change * median;
			}
			/**if (icp > 0) {
				ostringstream repeatMSG;
				repeatMSG << "Repeating ICP with a defined change, leafsize, and normal radius: " << change << ", " << icpLeafsize << ", and " << icpNormals;
				Log(repeatMSG.str());
			}

			if (icpLeafsize < 0 || icpNormals < 0) {
				Log("Leafsize or normal radius less than 0...skipping icp repetition");
				continue;

			}**/

			if (CONFIG.get<bool>("segment.segment", false)) {
				string polylineFile = CONFIG.get<string>("segment.icp_snip", "");
				if (polylineFile == "") {
					Log("No snip file provided. Continuing...");
					*filteredCloud = *simpleCloud2;
					*filteredReference = *simpleCloud1;
					/**UniformFilter(simpleCloud2, filteredCloud, icpLeafsize);
					UniformFilter(simpleCloud1, filteredReference, icpLeafsize);**/
				} 
				else
				{
					*cloudCopy = *cloud2;
					*simpleCopy = *simpleCloud2;
					*referenceCopy = *cloud1;
					*simpleReferenceCopy = *simpleCloud1;
					cout << "Clouds before snip. Data: " << simpleCopy->points.size() << endl;
					Segment(cloudCopy, simpleCopy, polylineFile);
					Segment(referenceCopy, simpleReferenceCopy, polylineFile);
					*filteredCloud = *simpleCopy;
					*filteredReference = *simpleReferenceCopy;
					cout << "Clouds after snip. Data: " << simpleCopy->points.size() << endl;
			
					/**UniformFilter(simpleCopy, filteredCloud, icpLeafsize);
					UniformFilter(simpleCloud1, filteredReference, icpLeafsize);**/
					WriteCloud("snip.txt", cloudCopy);
				}
			}
			else {
				*filteredCloud = *simpleCloud2;
				*filteredReference = *simpleCloud1;
				/**UniformFilter(simpleCloud2, filteredCloud, icpLeafsize);
				UniformFilter(simpleCloud1, filteredReference, icpLeafsize);**/
			}
			

			//float resolution = CalculateResolution(filteredCloud);
			ostringstream normalMSG;
			//icpNormals = resolution * CONFIG.get<float>("icp_align.normal_radius", 40);
			/**float icpNormals = CONFIG.get<float>("icp_align.normal_radius", 40);
			if (CONFIG.get<bool>("icp_align.use_resolution", false)) {
				icpNormals = CalculateResolution(filteredReference) * CONFIG.get<float>("icp_align.normal_radius", 10);
			}
			else {
				icpNormals =  CONFIG.get<float>("icp_align.normal_radius", 1);
			}
			normalMSG << "Calculating Normals with radius: " << icpNormals;
			Log(normalMSG.str());
			normalReference = CalculateNormals(filteredReference, icpNormals);
			normalCloud = CalculateNormals(filteredCloud, icpNormals);**/
			IterativeAlign(cloud2, simpleCloud2, simpleCloud1, filteredReference, filteredCloud, normalReference, normalCloud, icpLeafsize, icpNormals, median, degree);
		}
		WriteCloud(outputFolder + "\\" + ftext, cloud2);
		if (i == 1) {
			mergedCloud->points = cloud2->points;
			mergedSimple->points = simpleCloud2->points;
		}
		else {
			mergedCloud->points.insert(mergedCloud->points.end(), cloud2->points.begin(), cloud2->points.end());
			mergedSimple->points.insert(mergedSimple->points.end(), simpleCloud2->points.begin(), simpleCloud2->points.end());
		}
	}
	PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr filteredReference(new PointCloud<PointXYZ>);
	PointCloud<Normal>::Ptr normalReference(new PointCloud<Normal>);
	PointCloud<Normal>::Ptr normalCloud(new PointCloud<Normal>);
	*filteredCloud = *mergedSimple;
	*filteredReference = *simpleCloud1;

	float icpLeafsize = CONFIG.get<float>("full_icp.leafsize", 0.1);
	UniformFilter(mergedSimple, filteredCloud, icpLeafsize);
	UniformFilter(simpleCloud1, filteredReference, icpLeafsize);

	float icpNormals = CONFIG.get<float>("full_icp.normals", 0.5);

	ostringstream normalMSG;
	normalMSG << "Calculating Normals with radius: " << icpNormals;
	Log(normalMSG.str());
	normalReference = CalculateNormals(filteredReference, icpNormals);
	normalCloud = CalculateNormals(filteredCloud, icpNormals);
	//IterativeAlign(mergedCloud, mergedSimple, filteredReference, filteredCloud, normalReference, normalCloud);
	WriteCloud(outputFolder + "\\" + writeFilePrefix + "_merged.txt", mergedCloud);

	// Subsample cloud

	if (CONFIG.get<bool>("subsample.subsample", false)) {
		PointCloud<PointXYZI>::Ptr intensity(new PointCloud<PointXYZI>);
		PointCloud<PointXYZRGB>::Ptr color(new PointCloud<PointXYZRGB>);
		float subsampleRadius = CONFIG.get<float>("subsample.subsample_radius", 0.02);
		for (size_t pts = 0; pts <= mergedCloud->points.size(); pts++) {
			PointXYZRGBI point = mergedCloud->points[pts];
			PointXYZRGB cp;
			PointXYZI ip;
			float x = point.x;
			float y = point.y;
			float z = point.z;
			float r = point.r;
			float g = point.g;
			float b = point.b;
			float i = point.i;
			cp.x = x;
			cp.y = y;
			cp.z = z;
			cp.r = r;
			cp.g = g;
			cp.b = b;

			ip.x = x;
			ip.y = y;
			ip.z = z;
			ip.intensity = i;

			intensity->points.push_back(ip);
			color->points.push_back(cp);
		}
		ostringstream colorMSG;
		colorMSG << "Uniform filtering with color: " << subsampleRadius;
		Log(colorMSG.str());
		Log("Original cloud size " + to_string((long long)color->size()));
		UniformSampling<PointXYZRGB> sorrgb;
		sorrgb.setInputCloud(color);
		sorrgb.setRadiusSearch(subsampleRadius);
		sorrgb.filter(*color);
		Log("Filtered cloud size " + to_string((long long)color->size()));

		ostringstream intensityMSG;
		intensityMSG << "Uniform filtering with intensity: " << subsampleRadius;
		Log(intensityMSG.str());
		Log("Original cloud size " + to_string((long long)intensity->size()));
		UniformSampling<PointXYZI> sori;
		sori.setInputCloud(intensity);
		sori.setRadiusSearch(subsampleRadius);
		sori.filter(*intensity);
		Log("Filtered cloud size " + to_string((long long)intensity->size()));

		string filename = outputFolder + "\\" + writeFilePrefix + "_merged_subsampled.txt";
		FILE * file;
		file = fopen(filename.c_str(), "w");
		if (!file)
		{
			Log("File could not be created.");
			return(-1);
		}
		fprintf(file, "\\X Y Z I R G B\n");

		for (size_t f = 0; f <= intensity->points.size(); f++) {
			PointXYZRGB cp = color->points[f];
			PointXYZI ip = intensity->points[f];
			float x = cp.x;
			float y = cp.y;
			float z = cp.z;
			float r = cp.r;
			float g = cp.g;
			float b = cp.b;
			float intensity = ip.intensity;
			fprintf(file, "%.5f,%.5f,%.5f,%.5f,%.0f,%.0f,%.0f\n", x, y, z, intensity, r, g, b);
		}
	}

	return (0);
}

PointCloud<PointXYZ>::Ptr CalculateKeypoints(PointCloud<PointXYZ>::Ptr &cloud)
{
	PointCloud<PointXYZ>::Ptr keypoints(new PointCloud<PointXYZ>);
	search::KdTree<PointXYZ>::Ptr tree = search::KdTree<PointXYZ>::Ptr(new search::KdTree<PointXYZ>);
	*keypoints = *cloud;

	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
	iss_detector.setSalientRadius(0.5);
	iss_detector.setNonMaxRadius(1);
	iss_detector.setInputCloud(cloud);
	iss_detector.setSearchMethod(tree);

	double resolution = CalculateResolution(cloud);
	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	iss_detector.setSalientRadius(5 * resolution);
	//cout << resolution << endl;
   // Set the radius for the application of the non maxima supression algorithm.
	iss_detector.setNonMaxRadius(3 * resolution);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	iss_detector.setMinNeighbors(5);
	// Set the upper bound on the ratio between the second and the first eigenvalue.
	iss_detector.setThreshold21(0.975);
	// Set the upper bound on the ratio between the third and the second eigenvalue.
	iss_detector.setThreshold32(0.975);
	// Set the number of prpcessing threads to use. 0 sets it to automatic.
	iss_detector.setNumberOfThreads(8);
	iss_detector.compute(*keypoints);

	return keypoints;
}

PointCloud<Normal>::Ptr CalculateNormals(PointCloud<PointXYZ>::Ptr &cloud, const float radius)
{
	// Create instance of the normal estimation class
	NormalEstimationOMP<PointXYZ, Normal> normalEstimation;
	normalEstimation.setViewPoint(1869, -27, 1839);
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

	ostringstream resMSG;
	resMSG << "Average resolution: " << resolution;
	Log(resMSG.str());

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
	PointCloud<Normal>::Ptr normalReferenceUpdated(new PointCloud<Normal>);
	PointCloud<PointXYZ>::Ptr referenceUpdated(new PointCloud<PointXYZ>);
	PointCloud<Normal>::Ptr normalCloudUpdated(new PointCloud<Normal>);
	PointCloud<PointXYZ>::Ptr cloudUpdated(new PointCloud<PointXYZ>);
	// Get keypoints
	for (int i = 0; i < (int)normalReference->size(); i++) {
		if (to_string(normalReference->points[i].normal_x) == "nan" || to_string(normalReference->points[i].normal_y) == "nan" || to_string(normalReference->points[i].normal_z) == "nan") {
			continue;
		}
		else {
			normalReferenceUpdated->push_back(normalReference->points[i]);
			referenceUpdated->push_back(filteredReference->points[i]);
		}
	}
	for (int i = 0; i < (int)normalCloud->size(); i++) {
		if (to_string(normalCloud->points[i].normal_x) == "nan" || to_string(normalCloud->points[i].normal_y) == "nan" || to_string(normalCloud->points[i].normal_z) == "nan") {
			continue;
		}
		else {
			normalCloudUpdated->push_back(normalCloud->points[i]);
			cloudUpdated->push_back(filteredCloud->points[i]);
		}
	}

	PointCloud<PointXYZ>::Ptr keypointsReference(new PointCloud<PointXYZ>);
	keypointsReference = CalculateKeypoints(referenceUpdated);
	PointCloud<PointXYZ>::Ptr keypointsCloud(new PointCloud<PointXYZ>);
	keypointsCloud = CalculateKeypoints(cloudUpdated);
	ostringstream keypoitsMSG;
	keypoitsMSG << "Keypoints Calculated" << endl;
	keypoitsMSG << "No of ISS reference keypoints: " << keypointsReference->size() << endl;
	keypoitsMSG << "No of ISS data keypoint: " << keypointsCloud->size();
	Log(keypoitsMSG.str());

	// Calculate feature histograms
	FPFHEstimationOMP<PointXYZ, Normal, FPFHSignature33> featureHistogram;
	featureHistogram.setNumberOfThreads(CONFIG.get<int>("feature_align.threads", 4));
	// Get features for the reference
	PointCloud<FPFHSignature33>::Ptr descriptorsReference(new PointCloud<FPFHSignature33>);
	search::KdTree<PointXYZ>::Ptr referenceTree(new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud(keypointsReference);
	featureHistogram.setInputNormals(normalReferenceUpdated);
	featureHistogram.setSearchMethod(referenceTree);
	featureHistogram.setRadiusSearch(feature);
	featureHistogram.setSearchSurface(referenceUpdated);
	featureHistogram.compute(*descriptorsReference);

	// Get features for the cloud to align
	PointCloud<FPFHSignature33>::Ptr descriptorsCloud(new PointCloud<FPFHSignature33>);
	search::KdTree<PointXYZ>::Ptr cloudTree(new search::KdTree<PointXYZ>);
	featureHistogram.setInputCloud(keypointsCloud);
	featureHistogram.setInputNormals(normalCloudUpdated);
	featureHistogram.setSearchMethod(cloudTree);
	featureHistogram.setSearchSurface(cloudUpdated);
	featureHistogram.compute(*descriptorsCloud);

	// Deterimine Correspondences
	boost::shared_ptr<Correspondences> correspondences(new Correspondences);
	boost::shared_ptr<Correspondences> remainingCorrespondences(new Correspondences);
	registration::CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> correspondenceEstimation;
	correspondenceEstimation.setInputSource(descriptorsCloud);
	correspondenceEstimation.setInputTarget(descriptorsReference);
	correspondenceEstimation.determineCorrespondences(*correspondences);
	ostringstream correspondencesMSG;
	correspondencesMSG << correspondences->size() << " correspondences found";
	Log(correspondencesMSG.str());

	Log("Rejecting outliers");
	registration::CorrespondenceRejectorSampleConsensus<PointXYZ> rejector;
	rejector.setInputSource(keypointsCloud);
	rejector.setInputTarget(keypointsReference);
	rejector.setInputCorrespondences(correspondences);
	rejector.setMaximumIterations(CONFIG.get<int>("feature_align.iterations", 400000));
	rejector.setInlierThreshold(inlier);
	rejector.setRefineModel(true);
	rejector.getRemainingCorrespondences(*correspondences, *remainingCorrespondences);
	ostringstream remainingMSG;
	remainingMSG << remainingCorrespondences->size() << " correspondences remaining";
	Log(remainingMSG.str());

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

void IterativeAlign(PointCloud<PointXYZRGBI>::Ptr &dataCloud, PointCloud<PointXYZ>::Ptr &simpleCloud, PointCloud<PointXYZ>::Ptr &simpleReference,
	PointCloud<PointXYZ>::Ptr filteredReference, PointCloud<PointXYZ>::Ptr filteredCloud,
	PointCloud<Normal>::Ptr normalReference, PointCloud<Normal>::Ptr normalCloud, float voxel, float normal, float median, float degree)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	*cloud1 = *filteredCloud;
	*cloud2 = *filteredReference;
	double resolution = CalculateResolution(filteredReference);
	cout << "Filtering ICP with voxel: " << voxel << endl;
	float normal_rad = CONFIG.get<float>("icp_align.normal_radius", 1);
	cout << "Normal radius: " << normal_rad << endl;
	UniformFilter(filteredCloud, cloud1, voxel);
	UniformFilter(filteredReference, cloud2, voxel);
	pcl::PointCloud<pcl::Normal>::Ptr normals2 = CalculateNormals(cloud2, normal_rad);
	pcl::PointCloud<pcl::Normal>::Ptr normals1 = CalculateNormals(cloud1, normal_rad);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud1_with_Normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud2_with_Normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr output_with_Normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::concatenateFields(*cloud1, *normals1, *cloud1_with_Normals);
	pcl::concatenateFields(*cloud2, *normals2, *cloud2_with_Normals);
	*output_with_Normals = *cloud1_with_Normals;
	*output = *cloud1;
	boost::shared_ptr<pcl::Correspondences> correspsondences(new pcl::Correspondences);
	Eigen::Matrix4f transformFinal(Eigen::Matrix4f::Identity());
	boost::shared_ptr<pcl::Correspondences> corresps(new pcl::Correspondences);
	boost::shared_ptr<pcl::Correspondences> corresps_filtered(new pcl::Correspondences);
	Eigen::Matrix4f transform_res_from_LM;
	int iteration; //Eigen::Matrix4f transform;
	pcl::registration::DefaultConvergenceCriteria<float> conv_crit(iteration, transform_res_from_LM, *corresps_filtered);
	//conv_crit.setMaximumIterationsSimilarTransforms (10);
	//conv_crit.setMaximumIterations(50);
	//conv_crit.setRotationThreshold (cos (0.5 * M_PI / 180.0));
	conv_crit.setAbsoluteMSE(0.000001);
	iteration = 0;
	do
	{
		iteration = iteration + 1;
		cout << "Iteration: " << iteration << endl;
		pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal> est;
		est.setInputSource(output_with_Normals);
		est.setInputTarget(cloud2_with_Normals);
		est.setSourceNormals(output_with_Normals);
		est.determineCorrespondences(*corresps);		cout << "Original correspondences: " << corresps->size() << endl;		registration::CorrespondenceRejectorOneToOne oneToOneRej;
		oneToOneRej.getRemainingCorrespondences(*corresps, *corresps_filtered);
		cout << "Correspondences left after one-to-one: " << corresps_filtered->size() << endl;

		pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr rej_normals(new pcl::registration::CorrespondenceRejectorSurfaceNormal);
		rej_normals->setInputCorrespondences(corresps_filtered);
		cout << "Using degree: " << degree << endl;
		double threshold = acos(deg2rad(degree));
		rej_normals->setThreshold(threshold);
		rej_normals->initializeDataContainer<pcl::PointNormal, pcl::PointNormal>();
		rej_normals->setInputSource<pcl::PointNormal>(output_with_Normals);
		rej_normals->setInputNormals<pcl::PointNormal, pcl::PointNormal>(cloud1_with_Normals);
		rej_normals->setInputTarget<pcl::PointNormal>(cloud2_with_Normals);
		rej_normals->setTargetNormals<pcl::PointNormal, pcl::PointNormal>(cloud2_with_Normals);
		rej_normals->getCorrespondences(*corresps_filtered);		cout << "Post surface correspondences: " << corresps_filtered->size() << endl;

		pcl::registration::CorrespondenceRejectorMedianDistance::Ptr rejector2(new pcl::registration::CorrespondenceRejectorMedianDistance);
		rejector2->setInputCorrespondences(corresps_filtered);
		cout << "Using median: " << median << endl;
		rejector2->setMedianFactor(median);
		rejector2->getCorrespondences(*corresps_filtered);		cout << "Post median correspondences: " << corresps_filtered->size() << endl;

		pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal, float> trans_est_lm;
		trans_est_lm.estimateRigidTransformation(*output_with_Normals, *cloud2_with_Normals, *corresps_filtered, transform_res_from_LM);
		transformFinal = transformFinal * transform_res_from_LM;

		pcl::transformPointCloud(*output, *output, transform_res_from_LM);
		pcl::transformPointCloudWithNormals(*output_with_Normals, *output_with_Normals, transform_res_from_LM);
	}
	while (!conv_crit.hasConverged());
	pcl::transformPointCloud(*dataCloud, *dataCloud, transformFinal);
	pcl::transformPointCloud(*simpleCloud, *simpleCloud, transformFinal);
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

	return (true);
}

void Segment(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud, string polylineFile) {
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


void SORFilter(PointCloud<PointXYZRGBI>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simpleCloud) {
	if (!CONFIG.get<bool>("sor.filter", false)) {
		return;
	}
	int meanK = CONFIG.get<float>("sor.meanK", 50);
	float stddev = CONFIG.get<float>("sor.stddev", 1.0);
	ostringstream sorMSG;
	sorMSG << "Performing SOR filtering with meanK and stddev: " << meanK << " and " << stddev;
	Log(sorMSG.str());
	pcl::StatisticalOutlierRemoval<PointXYZ> sorSimple;
	sorSimple.setInputCloud(simpleCloud);
	sorSimple.setMeanK(meanK);
	sorSimple.setStddevMulThresh(stddev);
	vector<int> indices;
	sorSimple.filter(indices);

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

void UniformFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float radius) {
	ostringstream colorMSG;
	colorMSG << "Uniform filtering with color RADIUS: " << radius;
	Log(colorMSG.str());
	Log("Original cloud size " + to_string((long long)cloud->size()));
	UniformSampling<PointXYZ> sorrgb;
	sorrgb.setInputCloud(cloud);
	sorrgb.setRadiusSearch(radius);
	sorrgb.filter(*filtered);
	Log("Filtered cloud size " + to_string((long long)filtered->size()));
	return;
}

void UniformFilter(const PointCloud<PointXYZRGB>::Ptr cloud, PointCloud<PointXYZRGB>::Ptr &filtered, float radius) {
	ostringstream colorMSG;
	colorMSG << "Uniform filtering with color: " << radius;
	Log(colorMSG.str());
	Log("Original cloud size " + to_string((long long)cloud->size()));
	UniformSampling<PointXYZRGB> sorrgb;
	sorrgb.setInputCloud(cloud);
	sorrgb.setRadiusSearch(radius);
	sorrgb.filter(*filtered);
	Log("Filtered cloud size " + to_string((long long)filtered->size()));
	return;
}

PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float lx, float ly, float lz)
{
	ostringstream leafMSG;
	leafMSG << "Voxel filtering with leafsize: " << lx;
	Log(leafMSG.str());
	Log("Original cloud size " + to_string((long long)cloud->size()));
	VoxelGrid<PointXYZ> sor3;
	sor3.setInputCloud(cloud);
	sor3.setLeafSize(lx, lx, lx);
	sor3.filter(*filtered);
	Log("Filtered cloud size " + to_string((long long)filtered->size()));;
	return filtered;
}

void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud) {
	FILE * file;
	file = fopen(filename.c_str(), "w");
	if (!file)
	{
		Log("File could not be created.");
		return;
	}
	fprintf(file, "\\X Y Z I R G B\n");
	for (size_t i = 0; i < cloud->points.size(); i++) {
		float x = cloud->points[i].x;
		float y = cloud->points[i].y;
		float z = cloud->points[i].z;
		float r = cloud->points[i].r;
		float g = cloud->points[i].g;
		float b = cloud->points[i].b;
		float intensity = cloud->points[i].i;

		fprintf(file, "%.5f,%.5f,%.5f,%.5f,%.0f,%.0f,%.0f\n", x, y, z, intensity, r, g, b);
	}
	fclose(file);;
	Log("Cloud written to " + filename);
	return;
}
