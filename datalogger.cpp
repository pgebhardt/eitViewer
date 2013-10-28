#include "datalogger.h"
#include <QDateTime>

DataLogger::DataLogger(QObject *parent) :
    QObject(parent) {
}

void DataLogger::add_data(Eigen::ArrayXXf data) {
    if (this->logging()) {
        // add current data set to buffer
        this->data().push_back(std::make_tuple(
            QDateTime::currentMSecsSinceEpoch(), data));
    }
}

void DataLogger::dump(std::ostream *ostream) {
    if (ostream == nullptr) {
        return;
    }

    for (const auto& d : this->data())
    for (mpFlow::dtype::index i = 0; i < std::get<1>(d).cols(); ++i) {
        (*ostream) << std::get<0>(d) << " " << std::get<1>(d).col(i).transpose() << std::endl;
    }
}
