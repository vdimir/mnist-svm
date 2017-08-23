
#ifndef TASK_SVM_H
#define TASK_SVM_H

#include <vector>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <map>
#include <memory>
#include <fstream>
#include <iterator>
#include <iostream>

using namespace std;

extern "C" {
    #include <vl/generic.h>
    #include <vl/svm.h>
}

class SvmModel {
    vector<double> w;
    double b;
    int label;

    SvmModel() : label(0), b(0) { }
public:
    template<typename It>
    SvmModel(int lab, It beg, It end, double bias) : label(lab), b(bias) {
        copy(beg,end,back_inserter(w));
    }

    static SvmModel* from_file(istream& is);

    void serialize(ostream& os);

    SvmModel* set_label(int lab) {
        label = lab;
        return this;
    }

    int get_label() {
        return label;
    }

    double predict(double* x) {
        double wx = 0;
        for (auto i = 0; i < w.size(); ++i) {
            wx += w[i] * x[i];
        }
        return wx - b;
    }
};


class Svm {
    VlSvm *svm;

    void train() {
        vl_svm_train(svm);
    }
public:
    Svm(VlSvmSolverType type,
        double const *data,
        vl_size dimension,
        vl_size numData,
        double const *labels,
        double lambda) {
        svm = vl_svm_new(type, data, dimension, numData, labels, lambda);
    }

    SvmModel* get_model(int label);

    ~Svm() {
        vl_svm_delete(svm);
    }
};


class SvmClassifier {
    vector<unique_ptr<SvmModel> > classifiers;
public:

    vector<pair<int, double> > predict_vec(double* x) {
        vector<pair<int, double> > res;
        for (auto& clf: classifiers) {
            res.push_back(make_pair( clf->get_label(), clf->predict(x)));
        }
        return res;
    }

    int predict(double* x) {
        auto res = predict_vec(x);
        auto mx = max_element(res.begin(), res.end(),
                    [](const std::pair<int, int>& left, const std::pair<int, int>& right){
                        return left.second <  right.second;
                    });
        return mx->first;
    }

    void add(SvmModel *pModel) {
        classifiers.push_back(unique_ptr<SvmModel>(pModel));
    }

    void serialize(ostream& os);
    static SvmClassifier* from_file(istream& is);
};


class SvmClassifierBuilder {
    vector<double> data;
    size_t dim;
    vector<int> labels;

    VlSvmSolverType type;
    double lambda;

    // encode each class using one-hot-encoding
    vector<double> one_hot_enc(int c);
public:
    SvmClassifierBuilder(VlSvmSolverType type,
                  double lambda) : lambda(lambda), type(type), dim(0) { }

    // create binary classificator for each class
    SvmClassifier* create_classifier();

    void set_dim(size_t n) {
        if  (dim == 0) {
            dim = n;
            return;
        }
        if (dim == n) {
            return;
        }
        throw invalid_argument("Data size cannot change");
    }

    void add_data(double* beg, int label) {
        if (dim == 0) {
            throw invalid_argument("Set data size first");
        }
        std::copy(beg, beg+dim, std::back_inserter(data));
        labels.push_back(label);
    }
};


#endif //TASK_SVM_H
