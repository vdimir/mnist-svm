
#ifndef TASK_FEATUREEXTRACTOR_H
#define TASK_FEATUREEXTRACTOR_H

#include <opencv2/core/core.hpp>
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
public:
    virtual result_type extract_feature(Mat& img) {
        HOGDescriptor hog(
                Size(20,20), //winSize
                Size(5,5), //blocksize
                Size(5,5), //blockStride,
                Size(5,5), //cellSize,
                8, //nbins,
                -1,
                0.2,
                true,
                64);

        vector<float> resf;
        hog.compute(img, resf);
        features.clear();
        copy(resf.begin(), resf.end(), back_inserter(features));
        return make_pair(&features[0], features.size());
    }

};
#endif //TASK_FEATUREEXTRACTOR_H
