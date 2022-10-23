#include<bits/stdc++.h>
using namespace std;

/* Clustering tool class */
class KMeansClustering {
private:
	/* Dataset */
	vector<pair<double, double>> data;

	/* Point-wise cluster information */
	vector<int> cluster_number;
	vector<pair<double, double>> centroid;
	vector<vector<double>> distance;
	
	/* Class variables */
	ofstream otp;
	int n, k;
	
	/* Calculates the Euclidean distance between two points */
	double dist(pair<double, double> p1, pair<double, double> p2) {
		double res = (p1.first - p2.first) * (p1.first - p2.first);
		res += (p1.second - p2.second) * (p1.second - p2.second);
		return sqrt(res);
	}

	/* Updates the distance matrix */
	void update_distances() {
		for(int i = 0; i < k; i++) {
			for(int j = 0; j < n; j++) {
				distance[i][j] = dist(centroid[i], data[j]);
			}
		}
	}

	/* Updates the cluster number vector */
	void update_cluster_numbers() {
		for(int i = 0; i < n; i++) {
			int cluster = 0;
			double min_dist = distance[0][i];
			for(int j = 0; j < k; j++) {
				if(min_dist > distance[j][i]) {
					min_dist = distance[j][i];
					cluster = j;
				}
			}
			cluster_number[i] = cluster;
		}
	}

	/* Updates the centroid vector and returns if any change was made */
	bool update_centroids() {
		bool change = false;
		vector<pair<double, double>> new_centroid(k, {0, 0});
		for(int i = 0; i < k; i++) {
			int num = 0;
			for(int j = 0; j < n; j++) {
				if(cluster_number[j] == i) {
					new_centroid[i].first += data[j].first;
					new_centroid[i].second += data[j].second;
					num++;
				}
			}
			new_centroid[i].first /= num;
			new_centroid[i].second /= num;
		}
		sort(new_centroid.begin(), new_centroid.end());
		if(new_centroid != centroid) {
			change = true;
			centroid = new_centroid;
		}
		return change;
	}

	/* Prints current information */
	void print_info(int iter_num) {
		otp<<"After iteration - "<<iter_num<<":\n";
		for(int i = 0; i < k; i++) {
			otp<<"Centroid of cluster-"<<(i + 1)<<": (";
			otp<<centroid[i].first<<", "<<centroid[i].second<<")\n";
		}
		for(int i = 0; i < n; i++) {
			otp<<"Cluster number of data point ("<<data[i].first;
			otp<<", "<<data[i].second<<"): "<<(cluster_number[i] + 1)<<"\n";
		}
		otp<<"\n";
	}

public:
	/* Constructor */
	KMeansClustering(vector<pair<double, double>> data, int k) {
		this->data = data;
		this->n = data.size();
		this->k = k;
		this->centroid = vector<pair<double, double>>(k);
		this->distance = vector<vector<double>>(k, vector<double>(n));
		this->cluster_number = vector<int>(n);
	}

	/* Clustering algorithm */
	void cluster(int max_iter) {
		otp.open("datasets/output.txt");
		set<int> initial_clusters;
		while(initial_clusters.size() < k) {
			int temp = (rand() % n);
			if(initial_clusters.count(temp)) continue;
			initial_clusters.insert(temp);
		}
		int index = 0;
		for(auto i : initial_clusters) {
			centroid[index++] = data[i];
		}
		sort(centroid.begin(), centroid.end());
		update_distances();
		update_cluster_numbers();
		print_info(1);
		for(int i = 1; i < max_iter; i++) {
			bool change = update_centroids();
			if(!change) break;
			update_distances();
			update_cluster_numbers();
			print_info(i + 1);
		}
		otp.close();
	}
};

/* Driver function */
int main() {
	/* Taking data input */
	vector<pair<double, double>> data;
	ifstream inp;
	inp.open("datasets/input.txt");
	int size; inp>>size;
	for(int i = 0; i < size; i++) {
		double x, y;
		inp>>x>>y;
		data.push_back({x, y});
	}
	int k; inp>>k;
	int max_iter; inp>>max_iter;

	/* Fitting the clustering tool on the data */
	KMeansClustering tool(data, k);
	tool.cluster(max_iter);
	inp.close();
}