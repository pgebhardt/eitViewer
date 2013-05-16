#ifndef MEASUREMENTSYSTEM_H
#define MEASUREMENTSYSTEM_H

#include <QObject>
#include <QThread>
#include <QUdpSocket>
#include <fasteit/fasteit.h>

class MeasurementSystem : public QObject
{
    Q_OBJECT
public:
    explicit MeasurementSystem(QObject *parent,
        std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> measurement);
    virtual ~MeasurementSystem();

public slots:
    void init();
    virtual void readyRead();

public:
    // accessors
    QUdpSocket& measurement_system_socket() { return *this->measurement_system_socket_; }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> measurement() {
        return this->measurement_;
    }
    void setMeasurementMatrix(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> value) {
        this->measurement_ = value;
    }
    QThread* thread() { return this->thread_; }

// member
private:
    QUdpSocket* measurement_system_socket_;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> measurement_;
    QThread* thread_;
};

#endif // MEASUREMENTSYSTEM_H
