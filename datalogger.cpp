#include "datalogger.h"
#include <QDateTime>

DataLogger::DataLogger(QObject *parent) :
    QObject(parent) {
}

void DataLogger::add_data(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> data) {
    if (this->logging()) {
        // copy data to host
        data->copyToHost(nullptr);
        cudaStreamSynchronize(nullptr);

        // add current data set to buffer
        this->data().push_back(std::make_tuple(
            QDateTime::currentMSecsSinceEpoch(),
            mpFlow::numeric::matrix::toEigen<mpFlow::dtype::real>(data)));
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
