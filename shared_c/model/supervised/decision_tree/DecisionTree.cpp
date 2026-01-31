#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

class DecisionTree {
public:
    enum class Impurity { Gini, Entropy, MSE, Variance };
    enum class FeatureType { Numerical, Categorical };

    DecisionTree(size_t max_depth = 5,
                 size_t min_samples_split = 2,
                 Impurity impurity = Impurity::Gini,
                 bool categorical_multiway = true)
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          impurity_(impurity),
          categorical_multiway_(categorical_multiway),
          feature_count_(0) {}

    // Train the tree; pass feature_types to mark categorical columns (default: all numerical).
    void fit(const vector<vector<double>>& X,
             const vector<double>& y,
             vector<FeatureType> feature_types = {}) {
        if (X.empty() || y.empty()) throw invalid_argument("Empty training data or labels");
        if (X.size() != y.size()) throw invalid_argument("Mismatched number of samples and labels");

        feature_count_ = X[0].size();
        if (feature_count_ == 0) throw invalid_argument("Training data has zero features");

        for (const auto& row : X) {
            if (row.size() != feature_count_) throw invalid_argument("Inconsistent feature sizes in X");
        }

        if (!feature_types.empty() && feature_types.size() != feature_count_) {
            throw invalid_argument("feature_types size must match number of features");
        }
        feature_types_ = feature_types;
        if (feature_types_.empty()) {
            feature_types_.assign(feature_count_, FeatureType::Numerical);
        }

        root_ = build_tree(X, y, 0);
    }

    // Predict the value/class for a single sample.
    double predict(const vector<double>& features) const {
        if (!root_) throw runtime_error("Model is not trained yet");
        if (features.size() != feature_count_) throw invalid_argument("Input feature size mismatch");

        const Node* node = root_.get();
        while (node && !node->is_leaf) {
            if (node->is_categorical) {
                if (node->is_multiway) {
                    auto it = node->children.find(features[node->feature_index]);
                    if (it != node->children.end()) {
                        node = it->second.get();
                    } else {
                        return node->prediction; // unseen category fallback
                    }
                } else {
                    if (features[node->feature_index] == node->category_value) {
                        node = node->left.get();
                    } else {
                        node = node->right.get();
                    }
                }
            } else {
                if (features[node->feature_index] <= node->threshold) {
                    node = node->left.get();
                } else {
                    node = node->right.get();
                }
            }
        }

        if (!node) throw runtime_error("Tree traversal failed; model may be corrupted");
        return node->prediction;
    }

    // Predict for a batch of samples.
    vector<double> predict(const vector<vector<double>>& X) const {
        vector<double> preds;
        preds.reserve(X.size());
        for (const auto& row : X) {
            preds.push_back(predict(row));
        }
        return preds;
    }

private:
    struct Node {
        bool is_leaf;
        bool is_categorical;
        bool is_multiway;
        size_t feature_index;
        double threshold;
        double category_value;
        double prediction;
        unique_ptr<Node> left;
        unique_ptr<Node> right;
        unordered_map<double, unique_ptr<Node>> children;

        explicit Node(double pred)
            : is_leaf(true),
              is_categorical(false),
              is_multiway(false),
              feature_index(0),
              threshold(0.0),
              category_value(0.0),
              prediction(pred) {}

        Node(size_t feature_idx, double thr, bool categorical, bool multiway, double cat_value, double fallback)
            : is_leaf(false),
              is_categorical(categorical),
              is_multiway(multiway),
              feature_index(feature_idx),
              threshold(thr),
              category_value(cat_value),
              prediction(fallback) {}
    };

    struct SplitInfo {
        bool found{false};
        size_t feature{0};
        double threshold{0.0};
        double category_value{0.0};
        bool is_categorical{false};
        bool is_multiway{false};
        double impurity{numeric_limits<double>::infinity()};
    };

    unique_ptr<Node> build_tree(const vector<vector<double>>& X,
                                const vector<double>& y,
                                size_t depth) {
        if (X.empty()) {
            return make_unique<Node>(0.0);
        }

        if (is_pure(y) || depth >= max_depth_ || X.size() < min_samples_split_) {
            return make_unique<Node>(leaf_value(y));
        }

        SplitInfo best = find_best_split(X, y);
        if (!best.found) {
            return make_unique<Node>(leaf_value(y));
        }

        const double fallback_pred = leaf_value(y);
        auto node = make_unique<Node>(best.feature,
                                      best.threshold,
                                      best.is_categorical,
                                      best.is_multiway,
                                      best.category_value,
                                      fallback_pred);

        if (best.is_categorical) {
            if (best.is_multiway) {
                unordered_map<double, vector<vector<double>>> child_X;
                unordered_map<double, vector<double>> child_y;
                for (size_t i = 0; i < X.size(); ++i) {
                    double key = X[i][best.feature];
                    child_X[key].push_back(X[i]);
                    child_y[key].push_back(y[i]);
                }
                for (auto& kv : child_X) {
                    node->children[kv.first] = build_tree(kv.second, child_y[kv.first], depth + 1);
                }
            } else { // one-vs-rest split
                vector<vector<double>> left_X;
                vector<vector<double>> right_X;
                vector<double> left_y;
                vector<double> right_y;

                for (size_t i = 0; i < X.size(); ++i) {
                    if (X[i][best.feature] == best.category_value) {
                        left_X.push_back(X[i]);
                        left_y.push_back(y[i]);
                    } else {
                        right_X.push_back(X[i]);
                        right_y.push_back(y[i]);
                    }
                }
                node->left = build_tree(left_X, left_y, depth + 1);
                node->right = build_tree(right_X, right_y, depth + 1);
            }
        } else {
            vector<vector<double>> left_X;
            vector<vector<double>> right_X;
            vector<double> left_y;
            vector<double> right_y;

            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][best.feature] <= best.threshold) {
                    left_X.push_back(X[i]);
                    left_y.push_back(y[i]);
                } else {
                    right_X.push_back(X[i]);
                    right_y.push_back(y[i]);
                }
            }
            node->left = build_tree(left_X, left_y, depth + 1);
            node->right = build_tree(right_X, right_y, depth + 1);
        }

        return node;
    }

    SplitInfo find_best_split(const vector<vector<double>>& X,
                              const vector<double>& y) const {
        SplitInfo best;
        for (size_t feature = 0; feature < feature_count_; ++feature) {
            SplitInfo candidate = (feature_types_[feature] == FeatureType::Numerical)
                                      ? eval_numerical_feature(feature, X, y)
                                      : eval_categorical_feature(feature, X, y);
            if (candidate.found && candidate.impurity < best.impurity) {
                best = candidate;
            }
        }
        return best;
    }

    SplitInfo eval_numerical_feature(size_t feature,
                                     const vector<vector<double>>& X,
                                     const vector<double>& y) const {
        SplitInfo best;
        const size_t m = X.size();
        if (m == 0) return best;

        vector<pair<double, size_t>> values(m);
        for (size_t i = 0; i < m; ++i) {
            values[i] = {X[i][feature], i};
        }
        sort(values.begin(), values.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t i = 1; i < m; ++i) {
            if (values[i - 1].first == values[i].first) continue;
            double threshold = 0.5 * (values[i - 1].first + values[i].first);

            vector<double> left_labels;
            vector<double> right_labels;
            left_labels.reserve(i);
            right_labels.reserve(m - i);

            for (size_t j = 0; j < m; ++j) {
                if (X[values[j].second][feature] <= threshold) {
                    left_labels.push_back(y[values[j].second]);
                } else {
                    right_labels.push_back(y[values[j].second]);
                }
            }

            if (left_labels.empty() || right_labels.empty()) continue;

            double weighted_impurity = impurity_numerical(left_labels, right_labels);
            if (weighted_impurity < best.impurity) {
                best.found = true;
                best.impurity = weighted_impurity;
                best.feature = feature;
                best.threshold = threshold;
                best.is_categorical = false;
                best.is_multiway = false;
            }
        }

        return best;
    }

    SplitInfo eval_categorical_feature(size_t feature,
                                       const vector<vector<double>>& X,
                                       const vector<double>& y) const {
        SplitInfo best;
        const size_t m = X.size();
        if (m == 0) return best;

        unordered_map<double, vector<double>> grouped;
        for (size_t i = 0; i < m; ++i) {
            grouped[X[i][feature]].push_back(y[i]);
        }

        if (grouped.size() <= 1) return best;

        if (categorical_multiway_) {
            double weighted = impurity_categorical_multiway(grouped);
            best.found = true;
            best.impurity = weighted;
            best.feature = feature;
            best.is_categorical = true;
            best.is_multiway = true;
        } else { // binary features
            for (const auto& kv : grouped) {
                vector<double> left_labels = kv.second; // equals this category
                vector<double> right_labels;
                right_labels.reserve(m - kv.second.size());

                for (const auto& other : grouped) {
                    if (other.first == kv.first) continue;
                    right_labels.insert(right_labels.end(), other.second.begin(), other.second.end());
                }

                if (left_labels.empty() || right_labels.empty()) continue;

                double weighted = impurity_categorical_binary(left_labels, right_labels);

                if (weighted < best.impurity) {
                    best.found = true;
                    best.impurity = weighted;
                    best.feature = feature;
                    best.is_categorical = true;
                    best.is_multiway = false;
                    best.category_value = kv.first;
                }
            }
        }

        return best;
    }

    bool is_pure(const vector<double>& labels) const {
        if (labels.empty()) return true;
        double first = labels.front();
        double tol = is_classification() ? 0.0 : 1e-9;
        for (double v : labels) {
            if (std::fabs(v - first) > tol) return false;
        }      
        return true;
    }

    double leaf_value(const vector<double>& labels) const {
        if (labels.empty()) return 0.0;
        if (is_classification()) {
            return majority_label(labels);
        }
        return mean(labels);
    }

    bool is_classification() const {
        return impurity_ == Impurity::Gini || impurity_ == Impurity::Entropy;
    }

    double majority_label(const vector<double>& labels) const {
        unordered_map<double, size_t> counts;
        for (double v : labels) {
            counts[v]++;
        }

        double best_label = labels.front();
        size_t best_count = 0;
        for (const auto& kv : counts) {
            if (kv.second > best_count) {
                best_label = kv.first;
                best_count = kv.second;
            }
        }
        return best_label;
    }

    double mean(const vector<double>& labels) const {
        if (labels.empty()) return 0.0;
        double sum = accumulate(labels.begin(), labels.end(), 0.0);
        return sum / static_cast<double>(labels.size());
    }

    // ----- Impurity helpers -----
    double gini_single(const vector<double>& labels) const {
        unordered_map<double, size_t> counts;
        for (double v : labels) counts[v]++;

        double impurity = 1.0;
        double inv_total = 1.0 / static_cast<double>(labels.size());
        for (const auto& kv : counts) {
            double p = kv.second * inv_total;
            impurity -= p * p;
        }
        return impurity;
    }

    double entropy_single(const vector<double>& labels) const {
        unordered_map<double, size_t> counts;
        for (double v : labels) counts[v]++;

        double inv_total = 1.0 / static_cast<double>(labels.size());
        double ent = 0.0;
        for (const auto& kv : counts) {
            double p = kv.second * inv_total;
            if (p > 0) {
                ent -= p * std::log2(p);
            }
        }
        return ent;
    }

    double mse_single(const vector<double>& labels) const {
        double m = mean(labels);
        double acc = 0.0;
        for (double v : labels) {
            double diff = v - m;
            acc += diff * diff;
        }
        return acc / static_cast<double>(labels.size());
    }

    double variance_single(const vector<double>& labels) const {
        return mse_single(labels);
    }

    double weighted_binary_impurity(const vector<double>& left,
                                    const vector<double>& right,
                                    double (DecisionTree::*single)(const vector<double>&) const) const {
        const double total = static_cast<double>(left.size() + right.size());
        if (total == 0.0) return 0.0;
        double imp_left = (this->*single)(left);
        double imp_right = (this->*single)(right);
        return (left.size() / total) * imp_left + (right.size() / total) * imp_right;
    }

    double weighted_multiway_impurity(const unordered_map<double, vector<double>>& groups,
                                      double (DecisionTree::*single)(const vector<double>&) const) const {
        double total = 0.0;
        for (const auto& kv : groups) total += kv.second.size();
        if (total == 0.0) return 0.0;

        double weighted = 0.0;
        for (const auto& kv : groups) {
            double imp = (this->*single)(kv.second);
            weighted += (static_cast<double>(kv.second.size()) / total) * imp;
        }
        return weighted;
    }

    // Numerical (binary) impurity by metric.
    double gini_numerical(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::gini_single);
    }
    double entropy_numerical(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::entropy_single);
    }
    double mse_numerical(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::mse_single);
    }
    double variance_numerical(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::variance_single);
    }

    // Categorical impurity: multiway aggregation.
    double gini_categorical_multiway(const unordered_map<double, vector<double>>& groups) const {
        return weighted_multiway_impurity(groups, &DecisionTree::gini_single);
    }
    double entropy_categorical_multiway(const unordered_map<double, vector<double>>& groups) const {
        return weighted_multiway_impurity(groups, &DecisionTree::entropy_single);
    }
    double mse_categorical_multiway(const unordered_map<double, vector<double>>& groups) const {
        return weighted_multiway_impurity(groups, &DecisionTree::mse_single);
    }
    double variance_categorical_multiway(const unordered_map<double, vector<double>>& groups) const {
        return weighted_multiway_impurity(groups, &DecisionTree::variance_single);
    }

    // Categorical impurity: binary (one-vs-rest) aggregation.
    double gini_categorical_binary(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::gini_single);
    }
    double entropy_categorical_binary(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::entropy_single);
    }
    double mse_categorical_binary(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::mse_single);
    }
    double variance_categorical_binary(const vector<double>& left, const vector<double>& right) const {
        return weighted_binary_impurity(left, right, &DecisionTree::variance_single);
    }

    // Metric dispatchers.
    double impurity_numerical(const vector<double>& left, const vector<double>& right) const {
        switch (impurity_) {
            case Impurity::Gini: return gini_numerical(left, right);
            case Impurity::Entropy: return entropy_numerical(left, right);
            case Impurity::MSE: return mse_numerical(left, right);
            case Impurity::Variance: return variance_numerical(left, right);
            default: return gini_numerical(left, right);
        }
    }

    double impurity_categorical_multiway(const unordered_map<double, vector<double>>& groups) const {
        switch (impurity_) {
            case Impurity::Gini: return gini_categorical_multiway(groups);
            case Impurity::Entropy: return entropy_categorical_multiway(groups);
            case Impurity::MSE: return mse_categorical_multiway(groups);
            case Impurity::Variance: return variance_categorical_multiway(groups);
            default: return gini_categorical_multiway(groups);
        }
    }

    double impurity_categorical_binary(const vector<double>& left, const vector<double>& right) const {
        switch (impurity_) {
            case Impurity::Gini: return gini_categorical_binary(left, right);
            case Impurity::Entropy: return entropy_categorical_binary(left, right);
            case Impurity::MSE: return mse_categorical_binary(left, right);
            case Impurity::Variance: return variance_categorical_binary(left, right);
            default: return gini_categorical_binary(left, right);
        }
    }

    size_t max_depth_;
    size_t min_samples_split_;
    Impurity impurity_;
    bool categorical_multiway_;
    size_t feature_count_;
    vector<FeatureType> feature_types_;
    unique_ptr<Node> root_;
};
