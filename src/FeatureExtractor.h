
#ifndef TASK_FEATUREEXTRACTOR_H
#define TASK_FEATUREEXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

#include "Svm.h"

class FeatureExtractor {

public:
    typedef pair<double*,size_t> result_type;
    virtual result_type extract_feature(Mat& img) = 0;
    virtual ~FeatureExtractor() {}
};


class RawImageExtractor : public FeatureExtractor {
    Mat res;
public:
    virtual result_type extract_feature(Mat& img) {
        img.convertTo(res, CV_64F, 1.0 / 255.0);
        return make_pair(res.ptr<double>(), res.total());
    }
};


class HogExtractor : public FeatureExtractor {
    vector<double> features;
    HOGDescriptor hog;
public:
    HogExtractor() : FeatureExtractor(),
                     hog(Size(20,20), Size(5,5), Size(5,5), Size(5,5), 8, -1, 0.2, true, 64)
    { }

    virtual result_type extract_feature(Mat& img) {
        Mat tmp;
        GaussianBlur(img, tmp, Size(5, 5), 1.2);
        vector<float> resf;
        hog.compute(tmp, resf);
        features.clear();
        copy(resf.begin(), resf.end(), back_inserter(features));
        return make_pair(&features[0], features.size());
    }

};
#endif //TASK_FEATUREEXTRACTOR_H
