#include<bits/stdc++.h>
using namespace std;

/* Compare function to sort the points before plotting */
bool compare(pair<double, double> &a, pair<double, double> &b) {
	if(a.second == b.second)
		return (a.first < b.first);
	return (a.second > b.second);
}

/* Linear regressor class */
class LinearRegression {
private:
	/* Dataset */
	vector<pair<double, double>> data;
	double slope, intercept;

public:
	/* Constructor */
	LinearRegression(vector<pair<double, double>> data) {
		this->data = data;
		this->slope = 0;
		this->intercept = 0;
	}

	/* Fits the regressor to the dataset */
	void fit() {
		int size = data.size();
		double x_mean = 0, y_mean = 0, xy_mean = 0, x2_mean = 0;
		for(auto p : this->data) {
			x_mean += p.first;
			y_mean += p.second;
			xy_mean += (p.first * p.second);
			x2_mean += (p.first * p.first);
		}
		x_mean /= size;
		y_mean /= size;
		xy_mean /= size;
		x2_mean /= size;
		double numerator = ((x_mean * xy_mean) - (y_mean * x2_mean));
		double denominator = ((x_mean * x_mean) - x2_mean);
		this->intercept = (numerator / denominator);
		this->slope = (y_mean - this->intercept) / x_mean;
	}

	/* Returns the predicted vector */
	vector<double> predict(vector<double> x) {
		vector<double> y_hat;
		for(auto xi : x)
			y_hat.push_back((this->slope * xi) + (this->intercept));
		return y_hat;
	}

	/* Returns the sqaured sum error on the given data */
	double squared_sum_error() {
		double res = 0;
		vector<double> y_hat;
		for(auto p : data)
			y_hat.push_back((this->slope * p.first) + (this->intercept));
		for(int i = 0; i < y_hat.size(); i++) {
			double temp = (this->data[i].second - y_hat[i]);
			temp *= temp;
			res += temp;
		}
		return res;
	}

	/* Prints the line equation */
	void line_equation() {
		cout<<"y = ";
		if(!this->intercept && !this->slope) {
			cout<<"0\n";
			return;
		}
		if(this->slope) {
			if (this->slope != 1 && this->slope != -1) cout << this->slope;
			else if (this->slope == -1) cout << "-";
			cout << "x";
		}
		if(this->intercept) {
			if(this->slope) {
				if (this->intercept < 0) cout << " - ";
				else cout << " + ";
			}
			cout << abs(this->intercept) << "\n";
		}
		else cout<<"\n";
	}

	/* Plots the data points */
	void plot() {
		auto temp = data;
		sort(temp.begin(), temp.end(), compare);
		int i = 0, n = temp.size(), y_coord = (int)temp[0].second;
		double max_x = -1;
		cout<<" ^\n"<<(y_coord--)<<"|";
		while(i < n) {
			max_x = max(max_x, temp[i].first);
			int steps = 1;
			while(steps < temp[i].first) {
				steps++;
				cout<<" ";
			}
			cout<<"*";
			steps++;
			i++;
			while(i < n && temp[i].second == temp[i - 1].second) {
				max_x = max(max_x, temp[i].first);
				while(steps < temp[i].first) {
					steps++;
					cout<<" ";
				}
				cout<<"*";
				steps++;
				i++;
			}
			if(i < n) {
				for(int j = 0; j < (temp[i - 1].second - temp[i].second); j++)
					cout<<"\n"<<(y_coord--)<<"|";
			}
		}
		while(y_coord > 0) cout<<"\n"<<(y_coord--)<<"|";
		cout<<"\n +";
		for(int j = 0; j < max_x; j++) cout<<"-";
		cout<<">\n  ";
		for(int j = 1; j <= max_x; j++) cout<<j;
		cout<<'\n';
	}
};

/* Driver function */
int main() {
	/* Taking data input */
	vector<pair<double, double>> data;
	cout<<"Enter the number of data points:";
	int size; cin>>size;
	cout<<"Enter the data:\n";
	for(int i = 0; i < size; i++) {
		double x, y;
		cin>>x>>y;
		data.push_back({x, y});
	}

	/* Initializing and fitting the regressor on the data */
	LinearRegression tool(data);
	tool.fit();

	/* Printing the line equation obtained */
	cout<<"Line Equation: ";
	tool.line_equation();

	/* Printing the squared sum error */
	cout<<"Squared Sum Error: "<<tool.squared_sum_error()<<'\n';

	/* Plotting the graph */
	cout<<"Plotted Graph:\n";
	tool.plot();

	/* Taking input for prediction */
	cout<<"Enter the number of x values to be predicted:";
	cin>>size;
	cout<<"Enter the x values:\n";
	vector<double> x_vector;
	double bin;
	for(int i = 0; i < size; i++) {
		cin>>bin;
		x_vector.push_back(bin);
	}

	/* Predicting and printing the values */
	vector<double> y_hat = tool.predict(x_vector);
	cout<<"Predicted values:\n";
	for(int i = 0; i < size; i++) cout<<x_vector[i]<<" : "<<y_hat[i]<<'\n';

	return 0;
}