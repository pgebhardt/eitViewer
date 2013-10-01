#include "measurementsystem.h"
#include <QDataStream>

MeasurementSystem::MeasurementSystem(QObject* parent) :
    QObject(parent), measurement_system_socket_(nullptr), buffer_pos_(0) {
    // create separat thread
    this->thread_ = new QThread(this);
    connect(this->thread(), SIGNAL(started()), this, SLOT(init()));
    this->moveToThread(this->thread());

    // init measurement buffer
    this->measurement_buffer_ = new std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>(1);
    this->measurement_buffer()[0] = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::real>>(1, 1, nullptr);

    this->thread()->start();
}

void MeasurementSystem::init() {
    // create udp socket
    this->measurement_system_socket_ = new QUdpSocket(this);
    this->measurement_system_socket().bind(3002, QUdpSocket::ShareAddress);
    connect(&this->measurement_system_socket(), SIGNAL(readyRead()), this, SLOT(readyRead()));
    this->time().restart();
}

void MeasurementSystem::readyRead() {
    // read measurement data for one excitation
    QByteArray datagram;
    datagram.resize((this->measurement_buffer()[this->buffer_pos()]->rows() + 1) * 8);
    this->measurement_system_socket().readDatagram(datagram.data(),
        datagram.size(), nullptr, nullptr);

    // extract measurement data
    QDataStream input_stream(datagram);
    double excitation = -1.0;
    input_stream >> excitation;
    if ((mpFlow::dtype::index)excitation < this->measurement_buffer()[this->buffer_pos()]->columns()) {
        double data = 0.0;
        for (mpFlow::dtype::index i = 0; i < this->measurement_buffer()[this->buffer_pos()]->rows(); ++i) {
            input_stream >> data;
            (*this->measurement_buffer()[this->buffer_pos()])(i, (mpFlow::dtype::index)excitation) = data;
        }
    }

    // move to next buffer element and upload current buffer element to gpu
    if ((mpFlow::dtype::index)excitation == this->measurement_buffer()[this->buffer_pos()]->columns() - 1) {
        // upload current buffer element to gpu
        this->measurement_buffer()[this->buffer_pos()]->copyToDevice(nullptr);

        // move to next buffer element
        this->buffer_pos() += 1;

        // emit data_ready signal when buffer is full
        if (this->buffer_pos() >= this->measurement_buffer().size()) {
            this->buffer_pos() = 0;

            emit this->data_ready(this->time().elapsed());
            this->time().restart();
        }
    }

}

