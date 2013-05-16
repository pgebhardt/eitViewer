#include "measurementsystem.h"
#include <QDataStream>

MeasurementSystem::MeasurementSystem(QObject *parent,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> measurement) :
    QObject(parent), measurement_system_socket_(nullptr), measurement_(measurement) {
    // create separat thread
    this->thread_ = new QThread();
    connect(this->thread(), SIGNAL(started()), this, SLOT(init()));
    this->moveToThread(this->thread());

    this->thread()->start();
}

MeasurementSystem::~MeasurementSystem() {

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
    if ((fastEIT::dtype::index)excitation < this->measurement()->columns()) {
        double data = 0.0;
        for (fastEIT::dtype::index i = 0; i < this->measurement()->rows(); ++i) {
            input_stream >> data;
            (*this->measurement())(i, (fastEIT::dtype::index)excitation) = data;
        }
    }
}
