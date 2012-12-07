#ifndef MEASUREMENTSYSTEM_H
#define MEASUREMENTSYSTEM_H

#include <QObject>
#include <QTcpSocket>
#include <fasteit/fasteit.h>

class MeasurementSystem : public QObject
{
    Q_OBJECT
public:
    explicit MeasurementSystem(QObject *parent = 0);
    virtual ~MeasurementSystem();
    void connectToSystem(const QHostAddress& address, int port);

    bool isConnected() { return this->measurement_system_socket().state() == QAbstractSocket::ConnectedState; }

signals:
    void error(QAbstractSocket::SocketError socket_error);
    
public slots:
    virtual void connected();
    virtual void readyRead();
    virtual void disconnected();
    virtual void connectionError(QAbstractSocket::SocketError socket_error);

public:
    // accessors
    const QTcpSocket& measurement_system_socket() const { return *this->measurement_system_socket_; }
    const fastEIT::dtype::size& electrodes_count() const { return this->electrodes_count_; }
    const fastEIT::dtype::size& drive_count() const { return this->drive_count_; }
    const fastEIT::dtype::size& measurement_count() const { return this->measurement_count_; }
    const fastEIT::Matrix<fastEIT::dtype::real>& voltage() const { return *this->voltage_; }

    // mutators
    QTcpSocket& measurement_system_socket() { return *this->measurement_system_socket_; }
    fastEIT::dtype::size& electrodes_count() { return this->electrodes_count_; }
    fastEIT::dtype::size& drive_count() { return this->drive_count_; }
    fastEIT::dtype::size& measurement_count() { return this->measurement_count_; }
    fastEIT::Matrix<fastEIT::dtype::real>& voltage() { return *this->voltage_; }

// member
private:
    QTcpSocket* measurement_system_socket_;
    fastEIT::dtype::size electrodes_count_;
    fastEIT::dtype::size drive_count_;
    fastEIT::dtype::size measurement_count_;
    fastEIT::Matrix<fastEIT::dtype::real>* voltage_;
};

#endif // MEASUREMENTSYSTEM_H
