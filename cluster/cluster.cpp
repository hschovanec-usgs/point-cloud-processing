#include <cmath>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
// PCL subsampling
#include <pcl/filters/voxel_grid.h>

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
using namespace Eigen;

property_tree::ptree CONFIG;

namespace pcl {
	struct PointCluster
	{
		PCL_ADD_POINT4D;
		int r;
		int g;
		int b;
		float i;
		float nx;
		float ny;
		float nz;
		float distance;
		int classifier;
		int cluster_number;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
	} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
}
POINT_CLOUD_REGISTER_POINT_STRUCT(PointCluster,
(float, x, x)
(float, y, y)
(float, z, z)
(float, i, i)
(int, r, r)
(int, g, g)
(int, b, b)
(float, nx, nx)
(float, ny, ny)
(float, nz, nz)
(float, distance, distance)
(float, classifier, classifier)
(int, cluster_number, cluster_number)
)

// Function headers
void ClusterVolumes(PointCloud<PointCluster>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simple, float epsilon, float minPoints);
bool LoadCloud(const string &filename, PointCloud<PointCluster> &cloud, PointCloud<PointXYZ> &simple);
property_tree::ptree LoadConfig(string configFile);
void Log(string message);
void RemoveLimit(PointCloud<PointCluster>::Ptr cloud, PointCloud<PointCluster>::Ptr &filtered, PointCloud<PointXYZ>::Ptr simple, PointCloud<PointXYZ>::Ptr &filteredSimple, string method);
void RemoveNoise(PointCloud<PointCluster> &cloud, PointCloud<PointCluster> &denoised, PointCloud<PointXYZ> &simple, PointCloud<PointXYZ> &denoiseSimple);
void SetupLogger();
void SeparateVolumes(PointCloud<PointCluster> cloud);
string ValidateConfig();
void WriteCloud(string filename, PointCloud<PointCluster> cloud);


template <typename ForwardIterator>
ForwardIterator remove_duplicates(ForwardIterator first, ForwardIterator last)
{
	auto new_last = first;

	for (auto current = first; current != last; ++current)
	{
		if (std::find(first, new_last, *current) == new_last)
		{
			if (new_last != current) *new_last = *current;
			++new_last;
		}
	}

	return new_last;
}


// Inline functions

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "A config file is required as the first parameter" << endl;
		return (-1);
	}

	string configFile = argv[1];
	CONFIG = LoadConfig(configFile);
	string isValidMessage = ValidateConfig();

	if (isValidMessage != "") {
		cout << "Invalid config parameters: " + isValidMessage << endl;
		return (-1);
	}
	SetupLogger();


	// Load reference cloud
	PointCloud<PointCluster>::Ptr cloud1(new PointCloud<PointCluster>);
	PointCloud<PointXYZ>::Ptr simple1(new PointCloud<PointXYZ>);
	PointCloud<PointCluster>::Ptr cloud2(new PointCloud<PointCluster>);
	PointCloud<PointXYZ>::Ptr simple2(new PointCloud<PointXYZ>);

	PointCloud<PointCluster>::Ptr filteredCloud(new PointCloud<PointCluster>);
	PointCloud<PointXYZ>::Ptr filteredSimple(new PointCloud<PointXYZ>);

	string cloudFile = CONFIG.get("first_cloud", "");
	Log("Loading cloud file: " + cloudFile);
	if (!LoadCloud(cloudFile, *cloud1, *simple1)) {
		Log("Unable to load file: " + cloudFile);
		return (-1);
	}
	string secondCloud = CONFIG.get("second_cloud", "");
	if (secondCloud != "") {
		Log("Loading cloud file: " + secondCloud);
		if (!LoadCloud(secondCloud, *cloud2, *simple2)) {
			Log("Unable to load file: " + secondCloud);
			return (-1);
		}
		PointCloud<PointCluster>::Ptr filteredCloud1(new PointCloud<PointCluster>);
		PointCloud<PointXYZ>::Ptr filteredSimple1(new PointCloud<PointXYZ>);
		PointCloud<PointCluster>::Ptr filteredCloud2(new PointCloud<PointCluster>);
		PointCloud<PointXYZ>::Ptr filteredSimple2(new PointCloud<PointXYZ>);
		cout << "1 size " << cloud1->points.size() << endl;
		cout << "2 size " << cloud2->points.size() << endl;
		RemoveLimit(cloud1, filteredCloud1, simple1, filteredSimple1, "negative");
		RemoveLimit(cloud2, filteredCloud2, simple2, filteredSimple2, "positive");
		filteredCloud->points = filteredCloud1->points;
		filteredSimple->points = filteredSimple1->points;
		filteredCloud->points.insert(filteredCloud->points.end(), filteredCloud2->points.begin(), filteredCloud2->points.end());
		filteredSimple->points.insert(filteredSimple->points.end(), filteredSimple2->points.begin(), filteredSimple2->points.end());
	}
	else {
		RemoveLimit(cloud1, filteredCloud, simple1, filteredSimple, "both");
	}


	Log("Clustering Volumes");
	string epsilon = CONFIG.get("epsilon", "2");
	float radius = atof(epsilon.c_str());
	float minimumPoints = CONFIG.get("minimum_points", 0);
	ClusterVolumes(filteredCloud, filteredSimple, radius, minimumPoints);

	Log("Saving cloud");
	vector< string > fileSplit;
	split(fileSplit, cloudFile, is_any_of("."), token_compress_on);
	string preDenoise = fileSplit[0] + "_clustered.txt";
	WriteCloud(preDenoise, *filteredCloud);

	if (CONFIG.get("output_individual_volumes", false) ) {
		Log("Separating volumes");
		SeparateVolumes(*filteredCloud);
	}

	return (0);
}


void ClusterVolumes(PointCloud<PointCluster>::Ptr &cloud, PointCloud<PointXYZ>::Ptr &simple, float epsilon, float minPoints) {
	int count = 0;
	int size = cloud->points.size();
	vector<bool> visited(size, false);
	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(simple);

	float percent = 1;
	for (int i = 0; i < size; i++) {
		if (ceil(100. * (float)i / (float)size) == percent) {
			cout << "Percentage Completed: " << percent << endl;
			percent += 1;
		}
		if (!visited[i]) {
			visited[i] = true;
			PointXYZ searchPoint = simple->points[i];
			vector<int> indices;
			vector<float> distances;
			int numFound = kdtree.radiusSearch(searchPoint, epsilon, indices, distances);
			if (numFound < minPoints) {
				cloud->points[i].cluster_number = -1;
			}
			else {
				int j = 0;
				count += 1;
				cloud->points[i].cluster_number = count;

				while (true) {
					int idx = indices[j];
					if (idx < 0 || idx > simple->points.size()) {
						// Do nothing
					}
					else {
						if (!visited[idx]) {
							visited[idx] = true;
							PointXYZ subPoint = simple->points[idx];
							vector<int> subIndices;
							vector<float> subDistances;
							int subFound = kdtree.radiusSearch(subPoint, epsilon, subIndices, subDistances);
							if (subFound > minPoints) {
								for (size_t kk = 0; kk < subIndices.size(); kk++) {
									if (std::find(indices.begin(), indices.end(), subIndices[kk]) == indices.end()) {
										// Index not already in the list
										indices.push_back(subIndices[kk]);
									}
								}
							}
						}
					}
					if (cloud->points[idx].cluster_number == 0) {
						cloud->points[idx].cluster_number = count;
					}
					j += 1;
					if (j >= indices.size()) {
						break;
					}
				}
			}

		}
	}
	cout << " FINISHED " << endl;
	return;
}


bool LoadCloud(const string &filename, PointCloud<PointCluster> &cloud, PointCloud<PointXYZ> &simple)
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

	int xidx, yidx, zidx, iidx, ridx, gidx, bidx, nxidx, nyidx, nzidx, distanceidx, classifieridx, clusteridx;
	xidx = CONFIG.get("value_order.x", 0);
	yidx = CONFIG.get("value_order.y", 1);
	zidx = CONFIG.get("value_order.z", 2);
	iidx = CONFIG.get("value_order.i", 3);
	ridx = CONFIG.get("value_order.r", 4);
	gidx = CONFIG.get("value_order.g", 5);
	bidx = CONFIG.get("value_order.b", 6);
	nxidx = CONFIG.get("value_order.nx", 7);
	nyidx = CONFIG.get("value_order.ny", 8);
	nzidx = CONFIG.get("value_order.nz", 9);
	distanceidx = CONFIG.get("value_order.distance", 10);
	classifieridx = CONFIG.get("value_order.classifier", 11);
	clusteridx = CONFIG.get("value_order.cluster_number", 12);


	while (!fs.eof())
	{
		getline(fs, line);
		// Ignore empty lines
		if (line == "" || line.at(0) == '/')
			continue;

		// Tokenize the line
		boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r, ,\t"), boost::token_compress_on);
		int r, g, b, cluster_number;
		float x, y, z, i, distance;

		PointCluster point;
		PointXYZ simplePoint;

		// x
		if (xidx >= st.size()) {
			point.x = 0;
		}
		else {
			point.x = float(atof(st[xidx].c_str()));
			simplePoint.x = float(atof(st[xidx].c_str()));
		}
		// y
		if (yidx >= st.size()) {
			point.y = 0;
		}
		else {
			point.y = float(atof(st[yidx].c_str()));
			simplePoint.y = float(atof(st[yidx].c_str()));
		}
		// z
		if (zidx >= st.size()) {
			point.z = 0;
		}
		else {
			point.z = float(atof(st[zidx].c_str()));
			simplePoint.z = float(atof(st[zidx].c_str()));
		}
		// i
		if (iidx >= st.size()) {
			point.i = 0;
		}
		else {
			point.i = float(atof(st[iidx].c_str()));
		}
		// r
		if (ridx >= st.size()) {
			point.r = 0;
		}
		else {
			point.r = int(atoi(st[ridx].c_str()));
		}
		// g
		if (gidx >= st.size()) {
			point.g = 0;
		}
		else {
			point.g = int(atoi(st[gidx].c_str()));
		}
		// b
		if (bidx >= st.size()) {
			point.b = 0;
		}
		else {
			point.b = int(atoi(st[bidx].c_str()));
		}
		// nx
		if (nxidx >= st.size()) {
			point.nx = 0;
		}
		else {
			point.nx = float(atof(st[nxidx].c_str()));
		}// ny
		if (nyidx >= st.size()) {
			point.ny = 0;
		}
		else {
			point.ny = float(atof(st[nyidx].c_str()));
		}// nz
		if (nzidx >= st.size()) {
			point.nz = 0;
		}
		else {
			point.nz = float(atof(st[nzidx].c_str()));
		}// distance
		if (distanceidx >= st.size()) {
			point.distance = 0;
		}
		else {
			point.distance = float(atof(st[distanceidx].c_str()));
		}// classifier
		if (classifieridx >= st.size()) {
			point.classifier = 0;
		}
		else {
			point.classifier = int(atoi(st[classifieridx].c_str()));
		}// cluster
		if (clusteridx >= st.size()) {
			point.cluster_number;
		}
		else {
			point.cluster_number = int(atoi(st[clusteridx].c_str()));
		}
		simple.push_back(simplePoint);
		cloud.push_back(point);
	}
	fs.close();

	return (true);
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

void RemoveLimit(PointCloud<PointCluster>::Ptr cloud, PointCloud<PointCluster>::Ptr &filtered, PointCloud<PointXYZ>::Ptr simple, PointCloud<PointXYZ>::Ptr &filteredSimple, string method) {
	if (!CONFIG.get("remove_LOD.remove", false)) {
		filtered = cloud;
		filteredSimple = simple;
		return;
	}

	PointCloud<PointCluster>::Ptr lodCloud(new PointCloud<PointCluster>);
	float min = CONFIG.get("remove_LOD.minimum", -0.01);
	float max = CONFIG.get("remove_LOD.maximum", 0.01);
	stringstream msg;
	if (method == "both") {
		msg << "Removing LOD between " << min << " and " << max << ".";

	}
	else if (method == "positive") {
		msg << "Removing LOD less than " << max << ".";
	}
	else if (method == "negative") {
		msg << "Removing LOD greater than " << min << ".";
	}
	Log(msg.str());
	bool stripVegetation = CONFIG.get("strip_to_bedrock", false);
	int bedrockValue = CONFIG.get("classifier", 1);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		float change = cloud->points[i].distance;
		float classifier = cloud->points[i].classifier;
		bool store = true;
		if (stripVegetation) {
			if (classifier != bedrockValue) {
				store = false;
			}
		}
		if (method == "both" && change <= max && change >= min) {
			store = false;
		}
		else if (method == "positive" && change <= max) {
			store = false;
		}
		else if (method == "negative" && change >= min) {
			store = false;
		}
		if (store) {
			lodCloud->points.push_back(cloud->points[i]);
			filteredSimple->points.push_back(simple->points[i]);
		}
	}
	cout << "cloud size " << lodCloud->points.size() << endl;
	cout << "simple size " << lodCloud->points.size() << endl;

	Log("Removing outliers");
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor(true);
	sor.setInputCloud(filteredSimple);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*filteredSimple);
	IndicesConstPtr ptindices = sor.getRemovedIndices();
	vector<int> indices = *ptindices;


	for (int j=0; j < lodCloud->points.size(); j++){
		if (std::find(indices.begin(), indices.end(), j) == indices.end()) {
			filtered->points.push_back(lodCloud->points[j]);
		}
	}

	return;
}

void RemoveNoise(PointCloud<PointCluster> &cloud, PointCloud<PointCluster> &denoised, PointCloud<PointXYZ> &simple, PointCloud<PointXYZ> &denoiseSimple) {
	if (!CONFIG.get("remove_noise", false)) {
		denoised = cloud;
		denoiseSimple = simple;
		return;
	}

	for (size_t i = 0; i < cloud.points.size(); i++) {
		float cluster = cloud.points[i].cluster_number;
		if (cluster != -1) {
			PointXYZ simplePoint = simple.points[i];
			PointCluster point = cloud.points[i];
			point.cluster_number = 0;
			denoised.push_back(point);
			denoiseSimple.push_back(simplePoint);
		}
	}
	return;
}

void SeparateVolumes(PointCloud<PointCluster> cloud) {
	Log("Filtering and separating volumes.");
	map<int, PointCloud<PointCluster>> clusters;
	for (size_t j = 0; j < cloud.points.size(); j++) {
		int cluster = cloud.points[j].cluster_number;
		PointCluster point = cloud.points[j];
		if (cluster != -1) {
			if (clusters.find(cluster) == clusters.end()) {
				PointCloud<PointCluster> clusterVec;
				clusterVec.push_back(point);
				clusters.insert(std::pair<int, PointCloud<PointCluster>>(cluster, clusterVec));
			}
			else {
				PointCloud<PointCluster> clusterVec = clusters[cluster];
				clusterVec.push_back(point);
				clusters[cluster] = clusterVec;
			}
		}
	}

	int counter = 1;

	bool filterChange = CONFIG.get("filter_one_way", false);
	bool classifyCluster = CONFIG.get("classify_cluster", false);
	if (classifyCluster) Log("Using cluster classification filtering.");
	if (filterChange) Log("Using cluster change filtering.");
	cout << "Number of Clusters before filtering "<< clusters.size() << endl;
	for (std::map<int, PointCloud<PointCluster>>::iterator iter = clusters.begin(); iter != clusters.end(); ++iter)
	{
		PointCloud<PointCluster> fall = iter->second;
		float min = CONFIG.get("minimum_threshold", 0);
		int bedrockNum = CONFIG.get("classifier", 1);
		if (classifyCluster) {
			Log("Filtering based on classification.");
			float clusterTol = CONFIG.get("cluster_tolerance", 0.4);
			int numOther = 0;
			for (size_t p = 0; p < fall.points.size(); p++) {
				float classNum = fall.points[p].classifier;
				if (classNum != bedrockNum && classNum != -1) {
					numOther += 1;
				}
			}
			if ((float)numOther / (float)fall.points.size() >= clusterTol) {
				continue;
			}
		}
		float change_difference = CONFIG.get("change_difference", 0.60);
		if (filterChange) {
			Log("Filtering based on change.");
			float hasPositive = 0;
			float hasNegative = 0;
			for (size_t p = 0; p < fall.points.size(); p++) {
				float change = fall.points[p].distance;
				if (change < 0) {
					hasNegative += 1;
				}
				else if (change > 0) {
					hasPositive += 1;
				}
			}
			if (hasPositive == 0 || hasNegative == 0) {
				cout << "Missing possitive or negative points." << endl;
				continue;
			}else if (abs(hasPositive - hasNegative)/(0.5*(hasNegative+hasPositive)) > change_difference)
			{
				cout << "Number of positive and negative change points have a " << change_difference * 100 << "% difference." << endl;
				continue;
			}
		}
		if (min != 0 && fall.points.size() < min) {
			Log("Filtering based on number of points.");
			continue;
		}
		string prefix = "fall__";
		if (counter < 10) {
			prefix += "00000";
		}
		else if (counter < 100) {
			prefix += "0000";
		}
		else if (counter < 1000) {
			prefix += "000";
		}
		else {
			prefix += "00";
		}
		string filename = prefix + to_string((long long)counter) + ".txt";
		counter += 1;
		string directory = CONFIG.get("output_directory", ".");
		string filePath = directory + "\\" + filename;
		FILE * file;
		file = fopen(filePath.c_str(), "w");
		if (!file)
		{
			Log("File " + filePath + " could not be created.");
			return;
		}
		for (size_t i = 0; i < fall.points.size(); i++) {
			float x = fall.points[i].x;
			float y = fall.points[i].y;
			float z = fall.points[i].z;
			float r = (float)fall.points[i].r;
			float g = (float)fall.points[i].g;
			float b = (float)fall.points[i].b;
			float dist = fall.points[i].distance;

			fprintf(file, "%.5f,%.5f,%.5f,%.5f,%.0f,%.0f,%.0f\n", x, y, z, dist, r, g, b);
		}
		fclose(file);
		Log("Cloud written to " + filePath);
	}
	Log(to_string((long long) counter - 1) + " volumes saved.");

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
			string cloudFile = CONFIG.get("first_cloud", "Cluster_Log.txt");
			vector<string> fileSplit;
			split(fileSplit, cloudFile, is_any_of("."), token_compress_on);
			string alignFilePrefix = fileSplit[0];
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
	string baseFile = CONFIG.get("first_cloud", "");
	if (baseFile == "") {
		isValidMessage += "\tcloud_file: An input cloud is required.\n";
	}

	string epsilon = CONFIG.get("epsilon", "");
	if (epsilon == "") {
		isValidMessage += "\tepsilon: An epsilon value required.\n";
	}

	string minPoints = CONFIG.get("minimum_points", "");
	if (minPoints == "") {
		isValidMessage += "\tminimum_points: A minimum points value required.\n";
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

void WriteCloud(string filename, PointCloud<PointCluster> cloud) {
	string directory = CONFIG.get("output_directory", ".");
	string filePath = directory + "\\" + filename;
	FILE * file;
	file = fopen(filePath.c_str(), "w");
	if (!file)
	{
		Log("File could not be created.");
		return;
	}
	if (CONFIG.get("header", true)){
		fprintf(file, "\\X Y Z I R G B\n");
	}
	for (size_t i = 0; i < cloud.points.size(); i++) {
		float x = cloud.points[i].x;
		float y = cloud.points[i].y;
		float z = cloud.points[i].z;
		int r = cloud.points[i].r;
		int g = cloud.points[i].g;
		int b = cloud.points[i].b;
		float intensity = cloud.points[i].i;
		float nx = cloud.points[i].nx;
		float ny = cloud.points[i].ny;
		float nz = cloud.points[i].nz;
		float change = cloud.points[i].distance;
		int cluster_number = cloud.points[i].cluster_number;

		fprintf(file, "%.5f,\t%.5f,\t%.5f,\t%.5f,\t%d,\t%d,\t%d,\t%.5f,\t%.5f,\t%.5f,\t%.5f,\t%d\n", x, y, z, intensity, r, g, b, nx, ny, nz, change, cluster_number);
	}
	fclose(file);
	Log("Cloud written to " + filePath);
	return;
}
