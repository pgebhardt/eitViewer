#ifndef MEASUREMENTSYSTEM_H
#define MEASUREMENTSYSTEM_H

#include <QObject>
#include <QThread>
#include <QUdpSocket>
#include <mpflow/mpflow.h>

class MeasurementSystem : public QObject {
    Q_OBJECT
public:
    explicit MeasurementSystem(QObject* parent=nullptr);

signals:
    void data_ready();

public slots:
    void init();
    virtual void readyRead();
    void setMeasurementBuffer(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* buffer) {
        this->measurement_buffer_ = buffer;
    }

public:
    // accessors
    QUdpSocket& measurement_system_socket() { return *this->measurement_system_socket_; }
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>& measurement_buffer() {
        return *this->measurement_buffer_;
    }
    QThread* thread() { return this->thread_; }
    size_t& buffer_pos() { return this->buffer_pos_; }

// member
private:
    QUdpSocket* measurement_system_socket_;
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* measurement_buffer_;
    QThread* thread_;
    size_t buffer_pos_;
};

#endif // MEASUREMENTSYSTEM_H
