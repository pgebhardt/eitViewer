#ifndef MEASUREMENTSYSTEM_H
#define MEASUREMENTSYSTEM_H

#include <QObject>
#include <QThread>
#include <QUdpSocket>
#include <QTime>
#include <mpflow/mpflow.h>

class MeasurementSystem : public QObject {
    Q_OBJECT
public:
    explicit MeasurementSystem(QObject* parent=nullptr);

signals:
    void data_ready(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<
        mpFlow::dtype::real>>>* data, int time_elapsed=0);

public slots:
    void init(mpFlow::dtype::index buffer_size, mpFlow::dtype::index rows,
        mpFlow::dtype::index columns);
    virtual void readyRead();

public:
    // accessors
    QUdpSocket& measurement_system_socket() { return *this->measurement_system_socket_; }
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>& measurement_buffer() {
        return *this->measurement_buffer_;
    }
    QThread* thread() { return this->thread_; }
    QTime& time() { return this->time_; }
    mpFlow::dtype::index& buffer_pos() { return this->buffer_pos_; }

// member
private:
    QUdpSocket* measurement_system_socket_;
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* measurement_buffer_;
    QThread* thread_;
    QTime time_;
    mpFlow::dtype::index buffer_pos_;
};

#endif // MEASUREMENTSYSTEM_H
