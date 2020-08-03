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
	struct PointChange
	{
		PCL_ADD_POINT4D;
		float nx;
		float ny;
		float nz;
		float distance;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
	} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
}
POINT_CLOUD_REGISTER_POINT_STRUCT(PointChange,
(float, x, x)
(float, y, y)
(float, z, z)
(float, nx, nx)
(float, ny, ny)
(float, nz, nz)
(float, distance, distance)
)

// Function headers

bool LoadPointLocations(const string &filename, PointCloud<PointXYZ> &cloud);
property_tree::ptree LoadConfig(string configFile);
void Log(string message);
void SetupLogger();
string ValidateConfig();
bool LoadCloudLocations(const string &filename, vector<string> &files, vector<string> &dates1, vector<string> &dates2);
bool LoadCloud(const string &filename, PointCloud<PointChange> &cloud, PointCloud<PointXYZ> &simple);


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

	/*-----------------------Get the cloud file--------------------------------*/
	vector<string> files;
	vector<string> firstDates;
	vector<string> secondDates;
	string cloudFiles = CONFIG.get("cloud_files", "");
	bool loaded = LoadCloudLocations(cloudFiles, files, firstDates, secondDates);
	if (!loaded) {
		Log("Unable to load file: " + cloudFiles);
		return (-1);
	 }
	/*-----------------------Get the point file--------------------------------*/
	string pointFile = CONFIG.get("point_file", "");
	PointCloud<PointXYZ> pointCloud;
	loaded = LoadPointLocations(pointFile, pointCloud);
	if (!loaded) {
		Log("Unable to load file: " + pointFile);
		return (-1);
	}
	/*-----------------------Get the difference--------------------------------*/
	PointXYZ searchPoint;
	string outdir = CONFIG.get("output_directory", "");
	string outFile = outdir + "\\" + CONFIG.get("output_file", "change.txt");
	FILE * file;
	file = fopen(outFile.c_str(), "w");
	if (!file)
	{
		Log("Point file (" + outFile + ") could not be created.");
		return (-1);
	}
	fprintf(file, "Header information will appear as columns in the following order:\n");
	fprintf(file, "0. x: The x position of the search point.\n");
	fprintf(file, "1. y: The y position of the search point.\n");
	fprintf(file, "2. z: The y position of the search point.\n");
	fprintf(file, "3. date1: The date 1 of the found point.\n");
	fprintf(file, "4. date2: The date 2 of the found point.\n");
	fprintf(file, "5. distance: Squared distance from the search point to the found point.\n");
	fprintf(file, "6. change: Change recorded from the found point.\n");
	fprintf(file, "7. nx: The x normal recorded from the found point.\n");
	fprintf(file, "8. ny: The y normal recorded from the found point.\n");
	fprintf(file, "9. nz: The z normal recorded from the found point.\n");
	fprintf(file, "10. mean_change: Mean change recorded from the K found points.\n");
	fprintf(file, "11. SD_change: SD change recorded from the K found points.\n");
	fprintf(file, "12. mean_nx: Mean nx recorded from the K found points.\n");
	fprintf(file, "13. SD_nx: SD nx recorded from the K found points.\n");
	fprintf(file, "14. mean_ny: Mean ny recorded from the K found points.\n");
	fprintf(file, "15. SD_ny: SD ny recorded from the K found points.\n");
	fprintf(file, "16. mean_nz: Mean nz recorded from the K found points.\n");
	fprintf(file, "17. SD_nz: SD nz recorded from the K found points.\n");
	fprintf(file, "X,\t Y,\t Z,\t date1,\t date2,\t distance,\t change,\t nx,\t ny,\t nz\n");
	cout << "Done printing" << endl;
	int K = CONFIG.get("K", 1);
	for (size_t f = 0; f < files.size(); f++) {
		//LOAD THE CLOUD TO SEARCH
		PointCloud<PointChange>::Ptr cloud(new PointCloud<PointChange>);
		PointCloud<PointXYZ>::Ptr simpleCloud(new PointCloud<PointXYZ>);
		Log("Loading cloud file: " + files[f]);
		if (!LoadCloud(files[f], *cloud, *simpleCloud)) {
			Log("Unable to load file: " + files[f]);
			return (-1);
		}
		for (size_t p = 0; p < pointCloud.points.size(); p++) {
			cout << "Point number: " << p << endl;
			searchPoint = pointCloud.points[p];
			searchPoint = pointCloud.points[p];
			pcl::search::KdTree<pcl::PointXYZ> tree;
			tree.setInputCloud(simpleCloud);
			int numberOfNeighbors = 0;
			vector<int> indices;
			vector<float> squaredDistances;
			numberOfNeighbors = tree.nearestKSearch(searchPoint, K, indices, squaredDistances);
			float x = searchPoint.x;
			float y = searchPoint.y;
			float z = searchPoint.z;
			PointChange foundPoint = cloud->points[indices[0]];
			float nx = foundPoint.nx;
			float ny = foundPoint.ny;
			float nz = foundPoint.nz;
			float change = foundPoint.distance;
			float distance = squaredDistances[0];
			string d1 = firstDates[f];
			string d2 = secondDates[f];
			float mean_change = 0;
			float mean_nx = 0;
			float mean_ny = 0;
			float mean_nz = 0;
			for (size_t m = 0; m < indices.size(); m++) {
				PointChange rangePoint = cloud->points[indices[m]];
				mean_change += rangePoint.distance;
				mean_nx += rangePoint.nx;
				mean_ny += rangePoint.ny;
				mean_nz += rangePoint.nz;
			}
			cout << "Calculating the Mean and SD of the " << indices.size() << " nearest points." << endl;
			mean_change /= indices.size();
			mean_nx /= indices.size();
			mean_ny /= indices.size();
			mean_nz /= indices.size();

			float sd_change = 0;
			float sd_nx = 0;
			float sd_ny = 0;
			float sd_nz = 0;
			for (size_t m = 0; m < indices.size(); m++) {
				PointChange rangePoint = cloud->points[indices[m]];
				sd_change += pow(rangePoint.distance - mean_change, 2);
				sd_nx += pow(rangePoint.nx - mean_nx, 2);
				sd_ny += pow(rangePoint.ny - mean_ny, 2);
				sd_nz += pow(rangePoint.nz - mean_nz, 2);
			}
			sd_change /= indices.size();
			sd_nx /= indices.size();
			sd_ny /= indices.size();
			sd_nz /= indices.size();
			
			sd_change = pow(sd_change, 0.5);
			sd_nx = pow(sd_nx, 0.5);
			sd_ny = pow(sd_ny, 0.5);
			sd_nz = pow(sd_nz, 0.5);

			fprintf(file, "%.5f,\t %.5f,\t %.5f,\t %s,\t %s,\t %.8f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f,\t %.5f\n", 
				x, y, z, d1.c_str(), d2.c_str(), distance, change, nx, ny, nz,
				mean_change, sd_change, mean_nx, sd_nx,
				mean_ny, sd_ny, mean_nz, sd_nz);
		}
	}
	fclose(file);
	


	/*-----------------------Write the results--------------------------------*/






	return (0);
}



bool LoadCloudLocations(const string &filename, vector<string> &files, vector<string> &dates1, vector<string> &dates2)
{
	ifstream fs;
	fs.open(filename.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail())
	{
		Log("Error opening file: " + filename);
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
		boost::split(st, line, boost::is_any_of("\t\r, ,\t"), boost::token_compress_on);

		if (3 != st.size()) {
			cout << "There must be three columns in the point file (date1, date2, filepath).\n";
			return (false);
		}

		string file_str = st[2];
		string date1 = st[0];
		string date2 = st[1];
		files.push_back(file_str);
		dates1.push_back(date1);
		dates2.push_back(date2);

	}
	fs.close();

	return (true);
}

bool LoadPointLocations(const string &filename, PointCloud<PointXYZ> &cloud) {
	ifstream fs;
	fs.open(filename.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail()) {
		Log("Error opening file: " + filename);
		fs.close();
		return (false);
	}

	string line;
	vector<string> st;

	while (!fs.eof()){
		getline(fs, line);
		// Ignore empty lines
		if (line == "" || line.at(0) == '/') {
			continue;
		}
		// Tokenize the line
		boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r, ,\t"), boost::token_compress_on);

		if (3 != st.size()) {
			cout << "There must be three columns in the point file (X, Y, Z).\n";
			return (false);
		}
		PointXYZ point;
		point.x = float(atof(st[0].c_str()));
		point.y = float(atof(st[1].c_str()));
		point.z = float(atof(st[2].c_str()));
		cloud.push_back(point);
	}
	fs.close();
	return (true);
}


bool LoadCloud(const string &filename, PointCloud<PointChange> &cloud, PointCloud<PointXYZ> &simple)
{
	ifstream fs;
	fs.open(filename.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail())
	{
		Log("Error opening file: " + filename);
		fs.close();
		return (false);
	}

	string line;
	vector<string> st;

	int xidx, yidx, zidx, nxidx, nyidx, nzidx, distanceidx;
	xidx = CONFIG.get("x", 0);
	yidx = CONFIG.get("y", 1);
	zidx = CONFIG.get("z", 2);
	nxidx = CONFIG.get("nx", 3);
	nyidx = CONFIG.get("ny", 4);
	nzidx = CONFIG.get("nz", 5);
	distanceidx = CONFIG.get("change", 6);


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

		PointChange point;
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
	string cloud_file = CONFIG.get("cloud_files", "");
	if (cloud_file == "") {
		isValidMessage += "\tcloud_files: A file listing the paths to each point cloud.\n";
	}

	string point_file = CONFIG.get("point_file", "");
	if (point_file == "") {
		isValidMessage += "\tpoint_file: File with list of points to examine. In form (X, Y, Z)\n";
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

	// File Structure
	string x = CONFIG.get("x", "");
	if (x == "") {
		isValidMessage += "\tx: The number of the x column.\n";
	}
	string y = CONFIG.get("y", "");
	if (y == "") {
		isValidMessage += "\ty: The number of the x column.\n";
	}
	string z = CONFIG.get("z", "");
	if (z == "") {
		isValidMessage += "\tz: The number of the x column.\n";
	}
	string nx = CONFIG.get("nx", "");
	if (nx == "") {
		isValidMessage += "\tnx: The number of the nx column.\n";
	}
	string ny = CONFIG.get("ny", "");
	if (ny == "") {
		isValidMessage += "\tny: The number of the nx column.\n";
	}
	string nz = CONFIG.get("nz", "");
	if (nz == "") {
		isValidMessage += "\tnz: The number of the nz column.\n";
	}
	string change = CONFIG.get("change", "");
	if (change == "") {
		isValidMessage += "\tchange: The number of the change column.\n";
	}

	return isValidMessage;
}
