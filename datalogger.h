#ifndef DATALOGGER_H
#define DATALOGGER_H

#include <QObject>
#include <vector>
#include <tuple>
#include <iostream>
#include <mpflow/mpflow.h>

class DataLogger : public QObject {
    Q_OBJECT
public:
    explicit DataLogger(QObject *parent = 0);
    
signals:
    
public slots:
    void start_logging() { this->logging() = true; }
    void stop_logging() { this->logging() = false; }
    void reset_log() { this->data().clear(); }
    void add_data(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> data);

public:
    void dump(std::ostream* ostream);

public:
    // Accessors
    bool& logging() { return this->logging_; }
    std::vector<std::tuple<qint64, Eigen::Array<mpFlow::dtype::real,
        Eigen::Dynamic, Eigen::Dynamic>>>& data() { return this->data_; }

private:
    bool logging_;
    std::vector<std::tuple<qint64, Eigen::Array<mpFlow::dtype::real,
        Eigen::Dynamic, Eigen::Dynamic>>> data_;
};

#endif // DATALOGGER_H
