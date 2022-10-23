#include<bits/stdc++.h>
using namespace std;

/* Decision tree classifier class */
class DecisionTreeClassifier {
private:
	/* Full dataset */
	vector<vector<string>> dataset;
	
	/* Compound node structure for both internal and leaf nodes */
	class DecisionTreeNode {
	public:
		/* Decision tree internal node structure */
		class DecisionTreeInternalNode {
		public:
			unordered_map<string, DecisionTreeNode*> children;
			int feature;

			/* Constructors */
			DecisionTreeInternalNode(int feature) : feature(feature) {}
			DecisionTreeInternalNode() : feature(-1) {}
		};
		
		/* Decision tree leaf node structure */
		class DecisionTreeLeafNode {
		public:
			string class_name;

			/* Constructors */
			DecisionTreeLeafNode(string  name) : class_name(std::move(name)) {}
			DecisionTreeLeafNode() = default;
		};
		
		bool is_leaf; /* Boolean to check if it is a leaf node */
		DecisionTreeInternalNode internal_node;
		DecisionTreeLeafNode leaf_node;

		/* Constructors */
		DecisionTreeNode(const DecisionTreeInternalNode& internal_node) {
			this->internal_node = internal_node;
			this->is_leaf = false;
		}
		DecisionTreeNode(const DecisionTreeLeafNode& leaf_node) {
			this->leaf_node = leaf_node;
			this->is_leaf = true;
		}
	};
	
	/* Defining internal and leaf node types */
	typedef DecisionTreeNode::DecisionTreeInternalNode Internal;
	typedef DecisionTreeNode::DecisionTreeLeafNode Leaf;

	/* Calculates the total information in the given tuples */
	double information(const set<int>& tuples) {
		set<string> unique_values;
		int size = (int)dataset[0].size();
		for(auto i : tuples) unique_values.insert(dataset[i][size - 1]);
		vector<double> probability(unique_values.size(), 0);
		int index = 0;
		for(const auto& v : unique_values) {
			for(auto i : tuples) {
				if(dataset[i][size - 1] == v)
					probability[index]++;
			}
			index++;
		}
		double sum = accumulate(probability.begin(), probability.end(), 0.0);
		for(auto& pi : probability)
			pi /= sum;
		double result = 0;
		for(const auto& pi : probability)
			result -= (pi * log2(pi));
		return result;
	}
	
	/* Calculates the entropy of a value of a feature with weight */
	double entropy_of_value(int feature, const string& value, const set<int>& tuples) {
		set<int> required_tuples;
		int count = 0;
		for(const auto& i : tuples) {
			if(dataset[i][feature] == value) {
				required_tuples.insert(i);
				count++;
			}
		}
		return (count * information(required_tuples));
	}
	
	/* Calculates the information of the feature */
	double leftover_info(int feature, const set<int>& tuples) {
		set<string> unique_values;
		for(auto i : tuples) unique_values.insert(dataset[i][feature]);
		int total = (int)tuples.size();
		double result = 0;
		for(const auto& v : unique_values)
			result += (entropy_of_value(feature, v, tuples) / total);
		return result;
	}

	/* Selects the best feature at an internal node */
	int select_best_feature(const set<int>& features, const set<int>& tuples) {
		double min_leftover_info = INT_MAX;
		int result = -1;
		for(const auto& f : features) {
			double current_leftover_info = leftover_info(f, tuples);
			if(min_leftover_info > current_leftover_info) {
				min_leftover_info = current_leftover_info;
				result = f;
			}
		}
		return result;
	}

	/* Checks if all the tuples belong to the same class */
	string has_impurity(const set<int>& tuples) {
		set<string> unique_values;
		int size = (int)dataset[0].size();
		for(auto i : tuples) unique_values.insert(dataset[i][size - 1]);
		return ((unique_values.size() > 1) ? "" : *unique_values.begin());
	}

	/* Returns the most frequent class in the remaining tuples */
	string majority_class(const set<int>& tuples) {
		unordered_map<string, int> frequency;
		int size = (int)dataset[0].size();
		for(auto i : tuples) frequency[dataset[i][size - 1]]++;
		int max_count = -1;
		string result;
		for(const auto& p : frequency) {
			if(p.second > max_count) {
				max_count = p.second;
				result = p.first;
			}
		}
		return result;
	}
	
	/* Returns a set containing the unique values of a feature */
	set<string> distinct_values(int feature, const set<int>& tuples) {
		set<string> result;
		for(const auto& i : tuples) result.insert(dataset[i][feature]);
		return result;
	}
	
	/* Recursive function to construct the tree */
	DecisionTreeNode* construct_recursive(set<int> features, const set<int>& tuples) {
		DecisionTreeNode* result;
		string check_impurity = has_impurity(tuples);
		if(!check_impurity.empty()) {
			result = new DecisionTreeNode(Leaf(check_impurity));
			return result;
		}
		if(features.empty()) {
			result = new DecisionTreeNode(Leaf(majority_class(tuples)));
			return result;
		}
		int best_feature = select_best_feature(features, tuples);
		result = new DecisionTreeNode(Internal(best_feature));
		features.erase(best_feature);
		set<string> dist_values = distinct_values(best_feature, tuples);
		for (const auto& v : dist_values) {
			set<int> child_tuples;
			for(auto i : tuples) {
				if(dataset[i][best_feature] == v)
					child_tuples.insert(i);
			}
			result->internal_node.children[v] = construct_recursive(features, child_tuples);
		}
		return result;
	}
	
	/* Recursive predict function */
	string predict_recursive(DecisionTreeNode* traversing_node, vector<string> features) {
		if(traversing_node->is_leaf)
			return traversing_node->leaf_node.class_name;
		int current_feature = traversing_node->internal_node.feature;
		Internal node = traversing_node->internal_node;
		return predict_recursive(node.children[features[current_feature]], features);
	}
	
	/* Decision tree head node */
	DecisionTreeNode* head_node{};
	
public:
	/* Constructor */
	DecisionTreeClassifier(const vector<vector<string>>& data) : dataset(data) {}
	
	/* Calls the recursive construct function */
	void construct() {
		int n = (int)dataset.size(), m = (int)dataset[0].size();
		set<int> tuples, features;
		for(int i = 1; i < n; i++)
			tuples.insert(i);
		for(int i = 0; i < m - 1; i++)
			features.insert(i);
		this->head_node = construct_recursive(features, tuples);
	}
	
	/* Predicts the class based on the given feature vector */
	string predict(const vector<string>& feature_vector) {
		return predict_recursive(head_node, feature_vector);
	}
	
	/* Prints the dataset */
	void print_data() {
		for(const auto& row : dataset) {
			for(const auto& word : row)
				cout<<word<<" ";
			cout<<'\n';
		}
	}
};

/* Driver function */
int main() {
	/* Taking input from a file */
	vector<vector<string>> data;
	ifstream inp("datasets/golf.csv");
	string temp;
	while(getline(inp, temp)) {
		vector<string> temp_vector;
		int size = (int)temp.length(), i = 0;
		while(i < size) {
			string bin;
			while(i < size && temp[i] != ',') {
				bin += temp[i];
				i++;
			}
			i++;
			temp_vector.push_back(bin);
		}
		data.push_back(temp_vector);
	}
	
	/* Fitting the decision tree classifier on the dataset */
	DecisionTreeClassifier tool(data);
	
	/* Construct the decision tree */
	tool.construct();
	
	/* Taking inputs for prediction */
	cout<<"\nEnter the number of test feature vectors: ";
	int tests; cin>>tests;
	cout<<"\nEnter the values of the features:\n";
	while(tests--) {
		vector<string> feature_vector;
		string value;
		cout<<'\n';
		for(int i = 0; i < (data[0].size() - 1); i++) {
			cout<<data[0][i]<<": ";
			cin>>value;
			feature_vector.push_back(value);
		}
		cout<<"Predicted - "<<data[0][data[0].size() - 1]<<": ";
		cout<<tool.predict(feature_vector)<<'\n';
	}
	inp.close();
	return 0;
}