#include "measurementsystem.h"
#include <QDataStream>

MeasurementSystem::MeasurementSystem(QObject* parent) :
    QObject(parent), measurement_system_socket_(nullptr), measurement_buffer_(nullptr),
    buffer_pos_(0) {
    // create separat thread
    this->thread_ = new QThread(this);
    this->moveToThread(this->thread());

    this->thread()->start();
}

void MeasurementSystem::init(mpFlow::dtype::index buffer_size, mpFlow::dtype::index rows,
    mpFlow::dtype::index columns) {
    // init measurement buffer
    if (this->measurement_buffer_ != nullptr) {
        delete this->measurement_buffer_;
        this->measurement_buffer_ = nullptr;
    }
    this->measurement_buffer_ = new std::vector<std::shared_ptr<mpFlow::numeric::Matrix<
        mpFlow::dtype::real>>>(buffer_size);
    for (mpFlow::dtype::index i = 0; i < buffer_size; ++i) {
        this->measurement_buffer()[i] = std::make_shared<mpFlow::numeric::Matrix<
            mpFlow::dtype::real>>(rows, columns, nullptr);
    }

    // create udp socket
    if (this->measurement_system_socket_ == nullptr) {
        this->measurement_system_socket_ = new QUdpSocket(this);
        this->measurement_system_socket().bind(3002, QUdpSocket::ShareAddress);
        connect(&this->measurement_system_socket(), SIGNAL(readyRead()), this, SLOT(readyRead()));
    }

    this->time().restart();
}

void MeasurementSystem::readyRead() {
    // read measurement data for one excitation
    QByteArray datagram;
    datagram.resize(this->measurement_buffer()[this->buffer_pos()]->rows() *
        this->measurement_buffer()[this->buffer_pos()]->columns() * 8);
    this->measurement_system_socket().readDatagram(datagram.data(),
        datagram.size(), nullptr, nullptr);

    // extract measurement data
    QDataStream input_stream(datagram);
    double data = 0.0;
    for (mpFlow::dtype::index row = 0; row < this->measurement_buffer()[this->buffer_pos()]->rows(); ++row)
    for (mpFlow::dtype::index column = 0; column < this->measurement_buffer()[this->buffer_pos()]->columns(); ++column) {
        input_stream >> data;
        (*this->measurement_buffer()[this->buffer_pos()])(row, column) = data;
    }

    // move to next buffer element and upload current buffer element to gpu
    this->buffer_pos() += 1;

    // emit data_ready signal when buffer is full
    if (this->buffer_pos() >= this->measurement_buffer().size()) {
        // upload measurement buffer to gpu
        for (mpFlow::dtype::index i = 0; i < this->measurement_buffer().size(); ++i) {
            this->measurement_buffer()[i]->copyToDevice(nullptr);
        }
        cudaStreamSynchronize(nullptr);

        // reset buffer pos
        this->buffer_pos() = 0;

        // emit signal for new data package ready
        emit this->data_ready(&this->measurement_buffer(), this->time().elapsed());
        this->time().restart();
    }
}

