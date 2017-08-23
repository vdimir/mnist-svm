#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

extern "C" {
#include <vl/generic.h>
#include <vl/svm.h>
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <cv.h>

using namespace cv;

#include "InputFileEntry.h"
#include "FeatureExtractor.h"
#include "Svm.h"
#include "common.h"

using namespace std;


void build_model(const string& inp, const string& out, FeatureExtractor&);
void run_model(const string &inp, const string &model_file, const string &out, FeatureExtractor&);

void cmd_args_guard(bool cond, const string& prog);

int main (int argc, const char * argv[]) {
    cmd_args_guard(argc > 2, argv[0]);

    RawImageExtractor ext;
    if (string("train").compare(argv[1]) == 0) {
        cmd_args_guard(argc == 4, argv[0]);

        build_model(argv[2], argv[3], ext);
        return 0;
    }
    if (string("test").compare(argv[1]) == 0) {
        cmd_args_guard(argc == 5, argv[0]);
        run_model(argv[2], argv[3], argv[4], ext);
        return 0;
    }
    cmd_args_guard(false, argv[0]);
    return 0;
}

// Read input file into vector and return true if success
template<typename T>
bool read_input_file(const string& fname, vector<T>& out) {
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        return false;
    }
    out.clear();
    std::copy(std::istream_iterator<T>(ifs),
              std::istream_iterator<T>(),
              std::back_inserter(out));
    return ifs.eof();
}

void cmd_args_guard(bool cond, const string& prog) {
    if (cond) { return; }
    cout << "Usage: " << prog << " train <input-file.txt> <output-file.dat>" << endl
         << "or: " << prog << " test <input-file.txt> <model.dat> <output-file.txt>"<< endl;
    exit(0);
}

Mat try_read_image(const string &fname) {
    Mat image;
    image = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
    ASSERT(image.data, "Could not open or find the image: " << fname);
    return image;
}

void build_model(const string &inp, const string &out, FeatureExtractor& ext) {
    vector<InputFileEntry> input_data;
    bool succ_read = read_input_file(inp, input_data);
    ASSERT(succ_read, "Error reading file: " << inp);

    cout << "Read data..." << endl;
    SvmClassifierBuilder clfb(VlSvmSolverSgd, 0.01);
    for (auto e: input_data) {
        Mat img = try_read_image(e.get_fname());
        auto data = ext.extract_feature(img);
        clfb.set_dim(data.second);
        clfb.add_data(data.first, e.get_label());
    }

    cout << "Processing " << input_data.size() << " images..." << endl;
    cout << "Start training..." << endl;
    auto clf = unique_ptr<SvmClassifier>(clfb.create_classifier());

    cout << "Saving model to " << out << "..." << endl;
    ofstream os(out, std::ios::binary);
    ASSERT(os.is_open(), "Error opening file: " << out);

    clf->serialize(os);
    os.close();
    cout << "Ok " << endl;
}


void run_model(const string &inp, const string &model_file, const string &out, FeatureExtractor& ext) {
    cout << "Read model from " << model_file << "..." << endl;
    ifstream is(model_file, std::ios::binary);
    ASSERT(is.is_open(), "Error reading file: " << model_file);
    auto clf = SvmClassifier::from_file(is);
    is.close();

    vector<InputFileEntry> input_data;
    bool succ_read = read_input_file(inp, input_data);
    ASSERT(succ_read, "Error reading file: " << inp);

    cout << "Processing "<< input_data.size() << " images..." << endl;

    ofstream os(out);
    ASSERT(os.is_open(), "Error opening file: " << out);

    cout << "Start recognition..." << endl;
    cout << "Write results to " << out << endl;

    os << "# FILENAME | ACTUAL | PREDICTION" << endl;
    for (auto& img_fname: input_data) {
        auto img = try_read_image(img_fname.get_fname());
        auto res = ext.extract_feature(img);
        int predicted_class = clf->predict(res.first);
        os << img_fname.get_fname() << " "
           << img_fname.get_label() << " "
           << predicted_class << endl;
    }
    os.flush();
    cout << "Ok" << endl;
}
