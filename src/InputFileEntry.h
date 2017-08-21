#ifndef TASK_ENTRY_H
#define TASK_ENTRY_H

#include <istream>
#include <ostream>
#include <string>

class InputFileEntry {
    std::string filename;
    int label;
public:
    InputFileEntry() : filename(), label(0) {}

    template<typename S>
    InputFileEntry(S fname, int lab) : filename(fname), label(lab) {}

    std::string get_fname() const {
        return filename;
    }

    int get_label() const {
        return label;
    }

    friend std::istream &operator>>(std::istream &, InputFileEntry &);

    friend std::ostream &operator<<(std::ostream &, const InputFileEntry &);
};

std::istream &operator>>(std::istream &is, InputFileEntry &f) {
    return is >> f.filename >> f.label;
}

std::ostream &operator<<(std::ostream &os, const InputFileEntry &f) {
    return os << f.filename << " " << f.label;
}

#endif //TASK_ENTRY_H
