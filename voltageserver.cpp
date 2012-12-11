#include "voltageserver.h"
#include <QDataStream>

VoltageServer::VoltageServer(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> voltage,
                             QObject *parent) :
    QTcpServer(parent), voltage_(voltage) {
    // create thread
    this->thread_ = new QThread();
    connect(this->thread(), SIGNAL(started()), this, SLOT(init()));
    this->moveToThread(this->thread());
}

void VoltageServer::init() {
    // listen on port 3000
    connect(this, SIGNAL(newConnection()), this, SLOT(acceptConnection()));
    this->listen(QHostAddress::Any, 3000);
}

void VoltageServer::acceptConnection() {
    // get client
    this->client_ = this->nextPendingConnection();

    // connect signals
    connect(this->client(), SIGNAL(readyRead()), this, SLOT(readyRead()));
    connect(this->client(), SIGNAL(disconnected()), this, SLOT(disconnected()));

    // close server
    this->close();
}

void VoltageServer::readyRead() {
    // create input stream
    QDataStream input_stream(this->client());

    // read matrix
    for (fastEIT::dtype::index column = 0; column < this->voltage()->columns(); ++column) {
        for (fastEIT::dtype::index row = 0; row < this->voltage()->rows(); ++row) {
            input_stream >> (*this->voltage())(row, column);
        }
    }
    this->voltage()->copyToDevice(NULL);

    // clear buffer
    this->client()->readAll();
}

void VoltageServer::diconnected() {
    // open server
    this->listen(QHostAddress::Any, 3000);
}
