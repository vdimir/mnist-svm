#include "Svm.h"

SvmClassifier *SvmClassifierBuilder::create_classifier() {
    SvmClassifier* classifiers = new SvmClassifier();
    auto dim = data.size() / labels.size();

    vector<int> classes;
    unique_copy(labels.cbegin(), labels.cend(), back_inserter(classes));
    for (auto c: classes) {
        auto lab_encode = one_hot_enc(c);
        Svm svm(type, &data[0], dim, labels.size(), &lab_encode[0], lambda);
        classifiers->add(svm.get_model(c));
    }
    return classifiers;
}

vector<double> SvmClassifierBuilder::one_hot_enc(int c) {
    vector<double> lab_encode;
    for (auto lab: labels) {
        lab_encode.push_back(lab == c ? 1 : -1);
    }
    return lab_encode;
}

SvmModel *Svm::get_model(int label) {
    train();
    auto w = vl_svm_get_model(svm) ;
    auto b = vl_svm_get_bias(svm) ;
    auto dim = vl_svm_get_dimension(svm);
    return new SvmModel(label, w, w+dim, b);
}

#define RW_RAW_DATA(method, data, n) (method(reinterpret_cast<char*>(&(data)), (n)*sizeof(data)))

SvmModel *SvmModel::from_file(istream &is) {
    auto model = new SvmModel();
    size_t sz;
    RW_RAW_DATA(is.read, model->label, 1);
    RW_RAW_DATA(is.read, model->b, 1);
    RW_RAW_DATA(is.read, sz, 1);
    model->w.resize(sz);
    RW_RAW_DATA(is.read, model->w[0], sz);
    return model;
}

void SvmModel::serialize(ostream &os) {
    auto sz = w.size();
    RW_RAW_DATA(os.write, label, 1);
    RW_RAW_DATA(os.write, b, 1);
    RW_RAW_DATA(os.write, sz, 1);
    RW_RAW_DATA(os.write, w[0], sz);
}

void SvmClassifier::serialize(ostream &os) {
    size_t sz = classifiers.size();
    RW_RAW_DATA(os.write, sz, 1);
    for (auto& c: classifiers) {
        c->serialize(os);
    }
}

SvmClassifier *SvmClassifier::from_file(istream &is) {
    size_t sz;
    RW_RAW_DATA(is.read, sz, 1);

    SvmClassifier* res = new SvmClassifier();
    for (int i = 0; i < sz; ++i) {
        SvmModel* m = SvmModel::from_file(is);
        res->add(m);
    }
    return res;
}
