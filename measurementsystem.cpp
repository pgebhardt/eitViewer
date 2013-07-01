#include "measurementsystem.h"
#include <QDataStream>

MeasurementSystem::MeasurementSystem(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> measurement,
    QObject* parent) :
    QObject(parent), measurement_system_socket_(nullptr), measurement_(measurement) {
    // create separat thread
    this->thread_ = new QThread(this);
    connect(this->thread(), SIGNAL(started()), this, SLOT(init()));
    this->moveToThread(this->thread());

    this->thread()->start();
}

void MeasurementSystem::init() {
    // create udp socket
    this->measurement_system_socket_ = new QUdpSocket(this);
    this->measurement_system_socket().bind(3002, QUdpSocket::ShareAddress);
    connect(&this->measurement_system_socket(), SIGNAL(readyRead()), this, SLOT(readyRead()));
}

void MeasurementSystem::readyRead() {
    // read measurement data for one excitation
    QByteArray datagram;
    datagram.resize((this->measurement()->rows() + 1) * 8);
    this->measurement_system_socket().readDatagram(datagram.data(),
        datagram.size(), nullptr, nullptr);

    // extract measurement data
    QDataStream input_stream(datagram);
    double excitation = -1.0;
    input_stream >> excitation;
    if ((mpFlow::dtype::index)excitation < this->measurement()->columns()) {
        double data = 0.0;
        for (mpFlow::dtype::index i = 0; i < this->measurement()->rows(); ++i) {
            input_stream >> data;
            (*this->measurement())(i, (mpFlow::dtype::index)excitation) = data;
        }
    }
    this->measurement()->copyToDevice(nullptr);
}
