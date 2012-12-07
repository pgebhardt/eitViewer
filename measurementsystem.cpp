#include "measurementsystem.h"
#include <QDataStream>

MeasurementSystem::MeasurementSystem(QObject *parent) :
    QObject(parent), measurement_system_socket_(NULL), electrodes_count_(0), drive_count_(0),
    measurement_count_(0), voltage_(NULL) {
    // create tcp socket
    this->measurement_system_socket_ = new QTcpSocket(this);

    // connect socket to slots
    connect(&this->measurement_system_socket(), SIGNAL(connected()), this, SLOT(connected()));
    connect(&this->measurement_system_socket(), SIGNAL(disconnected()), this, SLOT(disconnected()));
    connect(&this->measurement_system_socket(), SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(connectionError(QAbstractSocket::SocketError)));
}

MeasurementSystem::~MeasurementSystem() {
    // cleanup voltage
    if (this->voltage_ != NULL) {
        delete this->voltage_;
    }
}

void MeasurementSystem::connectToSystem(const QHostAddress& address, int port) {
    // connect to measurement system
    this->measurement_system_socket().connectToHost(address, port);
}

void MeasurementSystem::disconnectFromSystem() {
    // disconnect
    this->measurement_system_socket().disconnectFromHost();
}

void MeasurementSystem::connected() {
    // wait for data
    this->measurement_system_socket().waitForReadyRead(1000);

    // get electrodes, drive and measurement count
    QDataStream input_stream(&this->measurement_system_socket());
    input_stream >> this->electrodes_count();
    input_stream >> this->measurement_count();
    input_stream >> this->drive_count();

    // create matrix
    this->voltage_ = new fastEIT::Matrix<fastEIT::dtype::real>(this->measurement_count(),
                                                               this->drive_count(), NULL);

    // connect ready read signal
    connect(&this->measurement_system_socket(), SIGNAL(readyRead()), this, SLOT(readyRead()));
}

void MeasurementSystem::readyRead() {
    // create data stream
    QDataStream input_stream(&this->measurement_system_socket());

    // read voltage
    for (fastEIT::dtype::index column = 0; column < this->voltage().columns(); ++column) {
        for (fastEIT::dtype::index row = 0; row < this->voltage().rows(); ++row) {
            input_stream >> this->voltage()(row, column);
        }
    }

    // clear buffer
    this->measurement_system_socket().readAll();
}

void MeasurementSystem::disconnected() {
    // cleanup
    disconnect(&this->measurement_system_socket(), SIGNAL(readyRead()), this, SLOT(readyRead()));

    if (this->voltage_ != NULL) {
        delete this->voltage_;
        this->voltage_ = NULL;
    }
}

void MeasurementSystem::connectionError(QAbstractSocket::SocketError socket_error) {
    // emit error
    emit this->error(socket_error);
}
