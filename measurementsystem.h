#ifndef MEASUREMENTSYSTEM_H
#define MEASUREMENTSYSTEM_H

#include <QObject>
#include <QThread>
#include <QUdpSocket>
#include <mpflow/mpflow.h>

class MeasurementSystem : public QObject
{
    Q_OBJECT
public:
    explicit MeasurementSystem(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> measurement,
        QObject* parent=nullptr);

public slots:
    void init();
    virtual void readyRead();
    void setMeasurementMatrix(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> value) {
        this->measurement_ = value;
    }

public:
    // accessors
    QUdpSocket& measurement_system_socket() { return *this->measurement_system_socket_; }
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> measurement() {
        return this->measurement_;
    }
    QThread* thread() { return this->thread_; }

// member
private:
    QUdpSocket* measurement_system_socket_;
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> measurement_;
    QThread* thread_;
};

#endif // MEASUREMENTSYSTEM_H
