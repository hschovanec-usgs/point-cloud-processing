#include <cmath>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <limits>
#include <Eigen/Dense>

// PCL subsampling
#include <pcl/filters/voxel_grid.h>

// PCL IO
#include <pcl/point_types.h>

// Normals
#include <pcl/features/normal_3d_omp.h>

// Boost property tree
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

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
const double PI = 3.14159265;

namespace pcl{
struct PointXYZRGBI
{
	PCL_ADD_POINT4D;
	float r;
	float g;
	float b;
	float i;
	int cluster;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBI,
(float, x, x)
(float, y, y)
(float, z, z)
(float, r, r)
(float, g, g)
(float, b, b)
(float, i, i)
(int, cluster, cluster)
)

property_tree::ptree LoadConfig(string configFile) {
	property_tree::ptree pt;
	property_tree::read_json(configFile, pt);
	return pt;
}

float AverageDistance(vector<float> relativeDistances);
int CalculateDensity(const PointCloud<PointXYZ>::ConstPtr& cloud, float resolution);
PointCloud<Normal>::Ptr CalculateNormals(PointCloud<PointXYZ>::Ptr &cloud);
float Dot(Normal n, PointXYZ p);
PointCloud<pcl::PointXYZI>::Ptr FilterDistanceCalculation (int NN,
	PointCloud<PointXYZI>::Ptr Rawdist, KdTreeFLANN<pcl::PointXYZ> &kdtree_ref,
	PointCloud<PointXYZ>::Ptr cloud_ref);
float GetDistToAxis(Normal normal, PointXYZ delta, float distance);
PointXYZ GetVector(PointXYZ head, PointXYZ tail);
PointCloud<PointXYZI>::Ptr KMethod(const PointCloud<PointXYZ>::Ptr referenceCloud, const PointCloud<PointXYZ>::Ptr dataCloud,
	  			PointCloud<Normal>::Ptr normals);
bool LoadCloud(const string &filename, PointCloud<PointXYZRGBI> &cloud, PointCloud<PointXYZ> &simpleCloud);
void Log(string message);
PointCloud<PointXYZI>::Ptr ModifiedM3C2(const PointCloud<PointXYZ>::Ptr referenceCloud, const PointCloud<PointXYZ>::Ptr dataCloud,
	PointCloud<Normal>::Ptr normals);
Normal Normalize(Normal p);
void SetupLogger();
string ValidateConfig();
PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float rad);
void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud,
	PointCloud<PointXYZI>::Ptr distances, PointCloud<Normal>::Ptr normals);

string GetFilePrefix(string f) {
	vector< string > baseSplit;
	split(baseSplit, f, is_any_of("\\"), token_compress_on);
	string localFile = baseSplit.back();
	vector< string > fsplit;
	split(fsplit, localFile, is_any_of("."), token_compress_on);
	string fprefix = fsplit[0];
	return fprefix;
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
	string baselineFile = CONFIG.get("reference_file", "");
	string dataFile = CONFIG.get("data_file", "");
	Log("Loading base file: " + baselineFile);
	if (!LoadCloud(baselineFile, *cloud1, *simpleCloud1)) {
		Log("Unable to load file: " + baselineFile);
		return (-1);
	}

	Log("Loading base file: " + dataFile);
	PointCloud<PointXYZRGBI>::Ptr cloud2(new PointCloud<PointXYZRGBI>);
	PointCloud<PointXYZ>::Ptr simpleCloud2(new PointCloud<PointXYZ>);
	if (!LoadCloud(dataFile, *cloud2, *simpleCloud2)) {
		Log("Unable to load file: " + dataFile);
		return (-1);
	}

	string basefileprefix = GetFilePrefix(baselineFile);
	string cloudfileprefix = GetFilePrefix(dataFile);
	cout << "Base file Prefix: " << basefileprefix << endl;
	cout << "Data file Prefix: " << cloudfileprefix << endl;

	PointCloud<Normal>::Ptr normalReference = CalculateNormals(simpleCloud1);
	PointCloud<PointXYZI>::Ptr distances = ModifiedM3C2(simpleCloud1, simpleCloud2, normalReference);
	string fname = basefileprefix + "-" + cloudfileprefix + "_change.txt";
	WriteCloud(fname, cloud1, distances, normalReference);

	return (0);
}

float AverageDistance(vector<float> relativeDistances){
	float sumSize = 0;
	for (size_t i = 0; i < relativeDistances.size(); i++){
		if (!isnan(relativeDistances[i])) {
			sumSize += relativeDistances[i];
		}
	}
	return sumSize/(float)relativeDistances.size();
}

int CalculateDensity(const PointCloud<PointXYZ>::ConstPtr& cloud, float resolution){
	int density = 0;
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);
	int numberOfNeighbors = 0;

	for (int i = 0; i < cloud->size(); ++i)
	{
		vector<int> indices;
		vector<float> squaredDistances;
		numberOfNeighbors = tree.radiusSearch(i, resolution, indices, squaredDistances);
		density += numberOfNeighbors/resolution;
	}
	density /= cloud->size();
	return density;
}

PointCloud<Normal>::Ptr CalculateNormals(PointCloud<PointXYZ>::Ptr &cloud){
	/**
	Calculation of normals using svd on a set of subsampled points for
	a given radius.

	Args:
		cloud (PointCloud<PointXYZ>::Ptr): Reference point cloud.
		radius (const float): Radius to search and subsample points.

	Returns:
		PointCloud<Normal>::Ptr: Calculated normals

	Notes:
		Process follows:
			1) Get fit plane for entire cloud and project a normal.
				1a) Calculate the centroid of the cloud1
				1b) Compute svd for plane normals
			2) Loop through the points (pi)
				2a) Subsample the points around pi
				2b) Compute centroid of subsampled points
				2c) Compute svd for normals
				2d) Check that the normal calculated is pointing in
					the same direction as the plane normal
					(if not switch the orientation)
	**/
	float radius = CONFIG.get("normals.radius", 0.2);
	Normal normal;
	string method = CONFIG.get("normals.method", "");
	PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
	
	PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
	*filteredCloud = *cloud;
	if (CONFIG.get("normals.subsample_normals", false)) {
		filteredCloud = VoxelFilter(cloud, filteredCloud, CONFIG.get("normals.filter_radius", radius/2));
	}

	if (method == "svd"){

		PointXYZ planeNormal;
	    planeNormal.x = 1;
	    planeNormal.y = 0;
	    planeNormal.z = 0;

		// Calculating individual normals
		search::KdTree<PointXYZ> tree;
		tree.setInputCloud(filteredCloud);
		float percent = 0;
		ofstream myfile;
		for (size_t p = 0; p < cloud->points.size(); p++){
			if ( ceil(100. * (float) p / (float) cloud->points.size()) == percent){
				  cout << "Percentage Completed: " << percent << endl;
				  percent += 5;
			}
			int numberOfNeighbors;
			vector<int> indices;
			vector<float> squaredDistances;
			numberOfNeighbors = tree.radiusSearch(cloud->points[p], radius,
						indices, squaredDistances);
			MatrixXf subpoints(3, numberOfNeighbors);

			int minPoints = CONFIG.get("normals.min_points", 0);
			if (numberOfNeighbors >= minPoints || minPoints == 0){
				for (size_t s = 0; s < numberOfNeighbors; s++){
					subpoints(0, s) = cloud->points[indices[s]].x;
					subpoints(1, s) = cloud->points[indices[s]].y;
					subpoints(2, s) = cloud->points[indices[s]].z;
				}
				MatrixXf subcentroid = subpoints.rowwise().mean();
				subpoints.row(0).array() -= subcentroid(0);
			    subpoints.row(1).array() -= subcentroid(1);
			    subpoints.row(2).array() -= subcentroid(2);
				JacobiSVD<MatrixXf> subsvd(subpoints, ComputeFullU);
				MatrixXf sU = subsvd.matrixU();
				normal.normal_x = sU(0,2);
				normal.normal_y = sU(1,2);
				normal.normal_z = sU(2,2);
				float dot = Dot(normal, planeNormal);
				float magNormal = sqrt(pow(normal.normal_x, 2) + pow(normal.normal_y, 2) + pow(normal.normal_z, 2));
			    float magPlane = sqrt(pow(planeNormal.x, 2) + pow(planeNormal.y,2) + pow(planeNormal.z,2));
				float deg = acos(dot/(magNormal*magPlane)) * 180.0 / PI;
				if (abs(deg) > 90){
			      normal.normal_x *= -1;
			      normal.normal_y *= -1;
			      normal.normal_z *= -1;
			  }
		  }
		  else{
			  normal.normal_x = NAN;
			  normal.normal_y = NAN;
			  normal.normal_z = NAN;
		  }
			normals->points.push_back(normal);
		}
	}else{
		float vx, vy, vz;
		float min = std::numeric_limits<float>::min();
		float max = std::numeric_limits<float>::max();
		string viewpoint = CONFIG.get("normals.viewpoint", "+x");
		if (viewpoint == "+x"){
			vx = max;
			vy = 0;
			vz = 0;
		}else if (viewpoint == "-x"){
			vx = min;
			vy = 0;
			vz = 0;
		}else if (viewpoint == "+y"){
			vx = 0;
			vy = max;
			vz = 0;
		}else if (viewpoint == "-y"){
			vx = 0;
			vy = min;
			vz = 0;
		}else if (viewpoint == "+z"){
			vx = 0;
			vy = 0;
			vz = max;
		}else if (viewpoint == "-z"){
			vx = 0;
			vy = 0;
			vz = min;
		}else if (viewpoint == "origin"){
			vx = 0;
			vy = 0;
			vz = 0;
		}else if (viewpoint == "max"){
			vx = max;
			vy = max;
			vz = max;
		}else if (viewpoint == "min"){
			vx = min;
			vy = min;
			vz = min;
		}else{
			MatrixXf subpoints(3, cloud->points.size());
			int minPoints = CONFIG.get("normals.min_points", 0);
				for (size_t s = 0; s < cloud->points.size(); s++) {
					subpoints(0, s) = cloud->points[s].x;
					subpoints(1, s) = cloud->points[s].y;
					subpoints(2, s) = cloud->points[s].z;
				}
				MatrixXf subcentroid = subpoints.rowwise().mean();
				subpoints.row(0).array() -= subcentroid(0);
				subpoints.row(1).array() -= subcentroid(1);
				subpoints.row(2).array() -= subcentroid(2);
				JacobiSVD<MatrixXf> subsvd(subpoints, ComputeFullU);
				MatrixXf sU = subsvd.matrixU();
				vx = sU(0, 2)*10000;
				vy = sU(1, 2)*10000;
				vz = sU(2, 2)*10000;
		}
		Log("Viewpoint: " + viewpoint);
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
		normalEstimation.compute(*normals);
	}

	return normals;
}

float CalculateResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud){
	double resolution = 0.0;
	int numberOfPoints = 0;
	int numberOfNeighbors = 0;

	vector<int> indices(2);
	vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
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

float Dot(Normal n, PointXYZ p){
	float dot = n.normal_x * p.x + n.normal_y * p.y + n.normal_z * p.z;
	return dot;
}

PointCloud<pcl::PointXYZI>::Ptr FilterDistanceCalculation (int NN, PointCloud<PointXYZI>::Ptr Rawdist, KdTreeFLANN<pcl::PointXYZ> &kdtree_ref, PointCloud<PointXYZ>::Ptr cloud_ref)
		{
      std::vector<int> pointIdxNKNSearchNN(NN);
      std::vector<float> pointNKNSquaredDistanceNN(NN);

	    pcl::PointXYZ searchpoint;
		pcl::PointCloud<pcl::PointXYZI>::Ptr Filtdist (new pcl::PointCloud<pcl::PointXYZI>);
		Filtdist->width = Rawdist->points.size ();
		Filtdist->height = 1;
		Filtdist->points.resize (Filtdist->width * Filtdist->height);

		//#pragma omp parallel for
	  for (long ii = 0; ii < Rawdist->points.size (); ++ii)
	  {
		 searchpoint=cloud_ref->points[ii];
	     kdtree_ref.nearestKSearch (searchpoint, NN, pointIdxNKNSearchNN, pointNKNSquaredDistanceNN);
		 std::vector<float> Filt;
		//cout << "line 484" << endl;


		for (long jj = 0; jj < pointIdxNKNSearchNN.size (); ++jj)
		  {
			  if ((Rawdist->points[pointIdxNKNSearchNN[jj]].intensity)!=(Rawdist->points[pointIdxNKNSearchNN[jj]].intensity)) //if it is nan
			 {
			  //do nothing
			 }
			  else
			  {
				 Filt.push_back( Rawdist->points[pointIdxNKNSearchNN[jj]].intensity);
			  }

		  }

			 float averageNN;

		if (Filt.empty()){
			  averageNN= std::numeric_limits<double>::quiet_NaN();
		  }
		  else
		  {

			averageNN=AverageDistance(Filt);

			}

		Filtdist->points[ii].x=Rawdist->points[ii].x;
		Filtdist->points[ii].y=Rawdist->points[ii].y;
		Filtdist->points[ii].z=Rawdist->points[ii].z;
		Filtdist->points[ii].intensity=averageNN;
	  }

	  return Filtdist;}

	  float GetDistToAxis(Normal normal, PointXYZ delta, float distance){
	  	float x, y, z;
	  	x = delta.x - distance * normal.normal_x;
	  	y = delta.y - distance * normal.normal_y;
	  	z = delta.z - distance * normal.normal_z;
	  	return sqrt(x*x + y*y + z*z);
	  }

	  PointXYZ GetVector(PointXYZ head, PointXYZ tail){
	  	PointXYZ v;
	  	v.x = head.x - tail.x;
	  	v.y = head.y - tail.y;
	  	v.z = head.z - tail.z;
	  	return v;
	  }

	  Normal Normalize(Normal p){
	  	float norm = sqrt(p.normal_x * p.normal_x + p.normal_y * p.normal_y + p.normal_z * p.normal_z);

	  	p.normal_x /= norm;
	  	p.normal_y /= norm;
	  	p.normal_z /= norm;

	  	return p;
	  }

PointCloud<PointXYZI>::Ptr KMethod(const PointCloud<PointXYZ>::Ptr referenceCloud, const PointCloud<PointXYZ>::Ptr dataCloud,
	  			PointCloud<Normal>::Ptr normals){

	  	pcl::search::KdTree<pcl::PointXYZ> kdtree_data;
	  	kdtree_data.setInputCloud (dataCloud);
	  	double resolutiondata = CalculateResolution(dataCloud);
	  	// Set number of points to search for in data point cloud
	  	int minPoints = CONFIG.get("change.min_points", 6);
		float cylinderRadius = CONFIG.get("change.cylinder_radius", .2);
		float cylinderLength = CONFIG.get("change.cylinder_length", 10);
	  	pcl::PointXYZ searchpoint;
	  	std::vector<int> pointIdxNKNSearch(minPoints);
	  	std::vector<float> pointNKNSquaredDistance(minPoints);
	  	pcl::PointCloud<pcl::PointXYZ>::Ptr ShortDistVect (new pcl::PointCloud<pcl::PointXYZ>);
	  	ShortDistVect->width = minPoints;
	  	ShortDistVect->height = 1;
	  	ShortDistVect->points.resize (ShortDistVect->width * ShortDistVect->height);
	  	std::vector<double> RawdistMag(referenceCloud->points.size ());
	  	cout << "starting computation" << endl;
	  	pcl::PointCloud<pcl::PointXYZI>::Ptr Rawdist (new pcl::PointCloud<pcl::PointXYZI>);
	  	Rawdist->width = referenceCloud ->points.size ();
	  	Rawdist->height = 1;
	  	Rawdist->points.resize (Rawdist->width * Rawdist->height);

	  	//#pragma omp parallel for

	  	for (long i = 0; i < referenceCloud->points.size (); ++i)
	  	{
	  		// search point
	  		searchpoint = referenceCloud->points[i];
	  		kdtree_data.nearestKSearch (searchpoint, minPoints, pointIdxNKNSearch, pointNKNSquaredDistance);
	  		std::vector<float> dot_prod(minPoints);
	  		std::vector<float> cylinderRad;

	  		for (long j = 0; j < pointIdxNKNSearch.size (); ++j)
	  		{
	  			ShortDistVect->points[j].x= dataCloud->points[pointIdxNKNSearch[j]].x - referenceCloud->points[i].x;
	  			ShortDistVect->points[j].y=  dataCloud->points[pointIdxNKNSearch[j]].y - referenceCloud->points[i].y;
	  			ShortDistVect->points[j].z=  dataCloud->points[pointIdxNKNSearch[j]].z - referenceCloud->points[i].z;
	  			dot_prod[j] = Dot(normals->points[i], ShortDistVect->points[j]);

	  			// temp is the rejection
	  			float temp = sqrt(pow((ShortDistVect->points[j].x-normals->points[i].normal_x*dot_prod[j]),2)+pow((ShortDistVect->points[j].y-normals->points[i].normal_y*dot_prod[j]),2)+pow((ShortDistVect->points[j].z-normals->points[i].normal_z*dot_prod[j]),2));

	  			if (temp < 5*resolutiondata)
	  			{
	  				cylinderRad.push_back(dot_prod[j]);
	  			}

	  		}

	  		float average ;

	  		if (cylinderRad.empty()){
	  			average= std::numeric_limits<double>::quiet_NaN();
	  		}
	  		else
	  		{
	  			average=AverageDistance(cylinderRad);

	  			if (average<-cylinderLength)
	  			{
	  			average=std::numeric_limits<double>::quiet_NaN();
	  			}

	  			if (average>cylinderLength)
	  			{
	  			average=std::numeric_limits<double>::quiet_NaN();
	  			}
	  		}
	  		Rawdist->points[i].x=referenceCloud->points[i].x;
	  		Rawdist->points[i].y=referenceCloud->points[i].y;
	  		Rawdist->points[i].z=referenceCloud->points[i].z;
	  		Rawdist->points[i].intensity=average;
	  	}
	  	cout << "starting filtering" << endl;

	  	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	  	tree.setInputCloud (referenceCloud);
	  	pcl::PointCloud<pcl::PointXYZI>::Ptr Filtdist = FilterDistanceCalculation(minPoints, Rawdist, tree, referenceCloud);


	    return Filtdist;
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
	bool strip = CONFIG.get("strip_to_bedrock", false);
	int classifier = CONFIG.get("classifier", 2);

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
		else{
			int xidx = CONFIG.get("value_order.x", 0);
			int yidx = CONFIG.get("value_order.y", 1);
			int zidx = CONFIG.get("value_order.z", 2);
			int iidx = CONFIG.get("value_order.i", 9999);
			int ridx = CONFIG.get("value_order.r", 9999);
			int gidx = CONFIG.get("value_order.g", 9999);
			int bidx = CONFIG.get("value_order.b", 9999);
			int cidx = CONFIG.get("value_order.classifier", 9999);
			PointXYZ simplePoint;
			PointXYZRGBI point;
			if (ridx != 9999) {
				point.r = float(atof(st[ridx].c_str()));
			}
			else {
				point.r = 'nan';
			}
			if (ridx != 9999) {
				point.g = float(atof(st[gidx].c_str()));
			}
			else {
				point.g = 'nan';
			}
			if (ridx != 9999) {
				point.b = float(atof(st[bidx].c_str()));
			}
			else {
				point.b = 'nan';
			}
			if (ridx != 9999) {
				point.i = float(atof(st[iidx].c_str()));
			}
			else {
				point.i = 'nan';
			}
			if (ridx != 9999) {
				point.cluster = float(atof(st[cidx].c_str()));
			}
			else {
				point.cluster = 'nan';
			}

			if (strip &&  point.cluster != classifier ) {
				continue;
			}
			point.x = float(atof(st[xidx].c_str()));
			point.y = float(atof(st[yidx].c_str()));
			point.z = float(atof(st[zidx].c_str()));
			cloud.push_back(point);
			simplePoint.x = float(atof(st[xidx].c_str()));
			simplePoint.y = float(atof(st[yidx].c_str()));
			simplePoint.z = float(atof(st[zidx].c_str()));
			simpleCloud.push_back(simplePoint);
		}
		
	}
	fs.close();

	cloud.width = uint32_t(cloud.size()); cloud.height = 1; cloud.is_dense = true;
	return (true);
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

PointCloud<PointXYZI>::Ptr ModifiedM3C2(const PointCloud<PointXYZ>::Ptr referenceCloud, const PointCloud<PointXYZ>::Ptr dataCloud,
			PointCloud<Normal>::Ptr normals){
	/**
		Process:
			- Loop through each reference point.
				- Get the normal at that reference point.
				- Look around the reference point for points.
					- Find those points that are within the cylinder.
					- Calculate the distance along normal.
				- Check if a large number of points have been found. If they have CONTINUE TO NEXT REFERENCE POINT.
				- Start moving backward along the projection, and look around for points.
					- Find those points that are within the cylinder.
					- Calculate the distance along normal.
				- Check if a large number of points have been found. If they have CONTINUE TO NEXT REFERENCE POINT.
				- Start moving forward along the projection, and look around for points.
					- Find those points that are within the cylinder.
					- Calculate the distance along normal.
	**/

	// Defining cylinder variables
	int minPoints = CONFIG.get("change.min_points", 6);
	float cylinderRadius = CONFIG.get("change.cylinder_radius", .2);
	float cylinderLength = CONFIG.get("change.cylinder_length", 10);
	int numberSpheres = floor(2. * cylinderLength * sqrt(2.) / cylinderRadius);
	float movementLength = 2 * cylinderLength / numberSpheres;
	float directionalNumber = ceil((float) numberSpheres / 2.);
	cout <<"User defined minpoints: " << minPoints << endl;
	if (minPoints == -99999){
		float resolution = CalculateResolution(dataCloud);
		int density = CalculateDensity(dataCloud, 5*resolution);
		cout << density << endl;
		minPoints = ceil(4/3 * PI * pow(cylinderRadius, 2) * density);
	}
	cout <<"calculated minpoints: " << minPoints << endl;

	// Initializing iterative variables
	PointXYZ refPoint;
	Normal normal;
	PointXYZ currentLocation;

	// Initializing distance cloud
	PointCloud<PointXYZI>::Ptr distances (new PointCloud<PointXYZI>);
	distances->width = referenceCloud->points.size();
	distances->height = 1;
	distances->points.resize (distances->width * distances->height);

	search::KdTree<PointXYZ> kdtree;
	kdtree.setInputCloud(dataCloud);

	// Starting iteration through reference points
	int percent = 5;
	bool satisfied = false;
	for (size_t i = 0; i < referenceCloud->points.size(); i++){
		//Print each percent of loop
		  if ( ceil(100. * (float) i / (float) referenceCloud->points.size()) == percent){
			  cout << "Percentage Completed: " << percent << endl;
			  percent += 5;
		}

		// getting reference point and normal
		refPoint = referenceCloud->points[i];
		normal = (normals->points[i]);
		
		// setting x,y,z data for the distance point
		distances->points[i].x = refPoint.x;
		distances->points[i].y = refPoint.y;
		distances->points[i].z = refPoint.z;
		if (isnan(normal.normal_x) || isnan(normal.normal_y) || isnan(normal.normal_z)) {
			distances->points[i].intensity = std::numeric_limits<double>::quiet_NaN();
			continue;
		}

		// initializing index and distance vectors
		vector<int> indexValues;
		vector<float> subsampledDistances;

		for (int b = 1; b <= directionalNumber; b++) {
			if (b == 1) {
				currentLocation.x = refPoint.x + -1 * movementLength * normal.normal_x;
				currentLocation.y = refPoint.y + -1 * movementLength * normal.normal_y;
				currentLocation.z = refPoint.z + -1 * movementLength * normal.normal_z;
			}
			else {
				currentLocation.x += -1 * movementLength * normal.normal_x;
				currentLocation.y += -1 * movementLength * normal.normal_y;
				currentLocation.z += -1 * movementLength * normal.normal_z;
			}

			vector<int> backPointsIdx;
			vector<float> backPointsDistances;
			kdtree.radiusSearch(currentLocation, cylinderRadius, backPointsIdx, backPointsDistances);
			for (size_t np = 0; np < backPointsIdx.size(); np++) {
				if (subsampledDistances.size() > minPoints) {
					continue;
				}
				// Make sure that the point is not already accounted for
				//if (true){
				PointXYZ foundPoint = dataCloud->points[backPointsIdx[np]];
				PointXYZ vectorBetween = GetVector(foundPoint, refPoint);
				float normalDistance = Dot(normal, vectorBetween);
				float perpendicularDistance = GetDistToAxis(normal, vectorBetween, normalDistance);
				if (abs(normalDistance) < cylinderLength && abs(perpendicularDistance) < cylinderRadius) {
					indexValues.push_back(backPointsIdx[np]);
					subsampledDistances.push_back(normalDistance);
				}
				else {
					subsampledDistances.push_back(std::numeric_limits<double>::quiet_NaN());
				}

			}
		}
		for (int b = 0; b <= directionalNumber; b++) {
			if (b == 0) {
				currentLocation.x = refPoint.x;
				currentLocation.y = refPoint.y;
				currentLocation.z = refPoint.z;
			}else if (b==1) {
				currentLocation.x = refPoint.x + movementLength * normal.normal_x;
				currentLocation.y = refPoint.y + movementLength * normal.normal_y;
				currentLocation.z = refPoint.z + movementLength * normal.normal_z;
			}
			else {
				currentLocation.x += movementLength * normal.normal_x;
				currentLocation.y += movementLength * normal.normal_y;
				currentLocation.z += movementLength * normal.normal_z;
			}

			vector<int> backPointsIdx;
			vector<float> backPointsDistances;
			kdtree.radiusSearch(currentLocation, cylinderRadius, backPointsIdx, backPointsDistances);
			for (size_t np = 0; np < backPointsIdx.size(); np++) {
				if (subsampledDistances.size() > minPoints) {
					continue;
				}
				// Make sure that the point is not already accounted for
				//if (true){
				PointXYZ foundPoint = dataCloud->points[backPointsIdx[np]];
				PointXYZ vectorBetween = GetVector(foundPoint, refPoint);
				float normalDistance = Dot(normal, vectorBetween);
				float perpendicularDistance = GetDistToAxis(normal, vectorBetween, normalDistance);
				if (abs(normalDistance) < cylinderLength && abs(perpendicularDistance) < cylinderRadius) {
					indexValues.push_back(backPointsIdx[np]);
					subsampledDistances.push_back(normalDistance);
				}
				else {
					subsampledDistances.push_back(std::numeric_limits<double>::quiet_NaN());
				}

			}
		}
		float direction = AverageDistance(subsampledDistances);
		float directionMultiplier;
		if (direction > 0) {
			directionMultiplier = 1;
		}
		else {
			directionMultiplier = -1;
		}

		// > 0 is forward, <0 is reverse
		// look for data points around reference points

		vector<float> relativeDistances;
		vector<int> nearPointsIdx;
		vector<float> nearPointsDistances;
		kdtree.radiusSearch(refPoint, cylinderRadius, nearPointsIdx, nearPointsDistances);
		for (size_t np = 0; np < nearPointsIdx.size(); np++) {
			if (relativeDistances.size() > minPoints) {
				continue;
			}
			// Make sure that the point is not already accounted for
			PointXYZ foundPoint = dataCloud->points[nearPointsIdx[np]];
			PointXYZ vectorBetween = GetVector(foundPoint, refPoint);
			float normalDistance = Dot(normal, vectorBetween);
			float perpendicularDistance = GetDistToAxis(normal, vectorBetween, normalDistance);
			if (abs(normalDistance) <= cylinderLength && abs(perpendicularDistance) <= cylinderRadius) {
				indexValues.push_back(nearPointsIdx[np]);
				relativeDistances.push_back(normalDistance);
			}
			else {
				relativeDistances.push_back(std::numeric_limits<double>::quiet_NaN());
			}

		}

		// looking for datapoints in the direction determined
		for (int b = 1; b <= directionalNumber; b++) {
			if (b == 1) {
				currentLocation.x = refPoint.x + directionMultiplier * movementLength * normal.normal_x;
				currentLocation.y = refPoint.y + directionMultiplier * movementLength * normal.normal_y;
				currentLocation.z = refPoint.z + directionMultiplier * movementLength * normal.normal_z;
			}
			else {
				currentLocation.x += directionMultiplier * movementLength * normal.normal_x;
				currentLocation.y += directionMultiplier * movementLength * normal.normal_y;
				currentLocation.z += directionMultiplier * movementLength * normal.normal_z;
			}

			vector<int> backPointsIdx;
			vector<float> backPointsDistances;
			kdtree.radiusSearch(currentLocation, cylinderRadius, backPointsIdx, backPointsDistances);
			for (size_t np = 0; np < backPointsIdx.size(); np++) {
				if (relativeDistances.size() > minPoints) {
					continue;
				}
				// Make sure that the point is not already accounted for
				//if (true){
				PointXYZ foundPoint = dataCloud->points[backPointsIdx[np]];
				PointXYZ vectorBetween = GetVector(foundPoint, refPoint);
				float normalDistance = Dot(normal, vectorBetween);
				float perpendicularDistance = GetDistToAxis(normal, vectorBetween, normalDistance);
				if (abs(normalDistance) < cylinderLength && abs(perpendicularDistance) < cylinderRadius) {
					indexValues.push_back(backPointsIdx[np]);
					relativeDistances.push_back(normalDistance);
				}
				else {
					relativeDistances.push_back(std::numeric_limits<double>::quiet_NaN());
				}

			}
		}
		distances->points[i].intensity = AverageDistance(relativeDistances);
	}
	return distances;
}

void SetupLogger() {
	bool log = CONFIG.get("logging.log", false);
	bool autoGenerate = CONFIG.get("logging.autogenerate_log_file", false);
	string logFile = CONFIG.get("logging.log_file", "log.log");
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
			fname = "change.log";
		}
		else {
			fname = logFile;
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
	string baseFile = CONFIG.get("reference_file", "");
	if (baseFile == "") {
		isValidMessage += "\treference_file: A base file for alignment is required.\n";
	}
	string alignFilePrefix = CONFIG.get("data_file", "");
	if (alignFilePrefix == "") {
		isValidMessage += "\data_file: A data file for alignment is required.\n";
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

PointCloud<PointXYZ>::Ptr VoxelFilter(const PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr &filtered, float rad)
{
	ostringstream leafMSG;
	leafMSG << "Voxel filtering with leafsize: " << rad;
	Log(leafMSG.str());
	Log("Original cloud size " + to_string((long long)cloud->size()));
	VoxelGrid<PointXYZ> sor3;
	sor3.setInputCloud(cloud);
	sor3.setLeafSize(rad, rad, rad);
	sor3.filter(*filtered);
	Log("Filtered cloud size " + to_string((long long)filtered->size()));;
	return filtered;
}

void WriteCloud(string filename, PointCloud<PointXYZRGBI>::Ptr cloud,
	PointCloud<PointXYZI>::Ptr distances, PointCloud<Normal>::Ptr normals) {
	string outdir = CONFIG.get("output_directory", "");
	string outFile = outdir + "\\" + filename;
	FILE * file;
	file = fopen(outFile.c_str(), "w");
	if (!file)
	{
		Log("File could not be created.");
		return;
	}

	fprintf(file, "\\X Y Z I R G B NX NY NZ Distance\n");
	for (size_t i = 0; i < cloud->points.size(); i++) {
		float x = distances->points[i].x;
		float y = distances->points[i].y;
		float z = distances->points[i].z;
		float r = cloud->points[i].r;
		float g = cloud->points[i].g;
		float b = cloud->points[i].b;
		float nx = normals->points[i].normal_x;
		float ny = normals->points[i].normal_y;
		float nz = normals->points[i].normal_z;
		float distance = distances->points[i].intensity;
		float intensity = cloud->points[i].i;
		string tt = ", ";
		if (isnan(nx) || isnan(ny) || isnan (nz) || isnan(distance)) {
			fprintf(file, "%.5f,\t%.5f,\t%.5f,\t%.5f,\t%.0f,\t%.0f,\t%.0f,\tnan,\tnan,\tnan,\tnan\n", x, y, z, intensity, r, g, b);
		}
		else {
		fprintf(file, "%.5f,\t%.5f,\t%.5f,\t%.5f,\t%.0f,\t%.0f,\t%.0f,\t%.5f,\t%.5f,\t%.5f,\t%.5f\n", x, y, z, intensity, r, g, b, nx, ny, nz, distance);
		}
	}

	fclose(file);
	Log("Cloud written to " + outFile);
	return;
}
