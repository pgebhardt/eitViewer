#include "voltageserver.h"

VoltageServer::VoltageServer(fastEIT::Matrix<fastEIT::dtype::real>* voltage,
                             QObject *parent) :
    QTcpServer(parent), voltage_(NULL), thread_(NULL), client_(NULL), input_stream_(NULL) {
    // create thread
    this->thread_ = new QThread();

    // connect signals
    connect(&this->thread(), SIGNAL(started()), this, SLOT(init()));

    // move to thread
    this->moveToThread(&this->thread());

    // save voltage
    this->voltage_ = voltage;
}

void VoltageServer::init() {
    // listen on port 3000
    connect(this, SIGNAL(newConnection()), this, SLOT(acceptConnection()));
    this->listen(QHostAddress::Any, 3000);
}

void VoltageServer::acceptConnection() {
    // get client
    this->client_ = this->nextPendingConnection();
    this->input_stream_ = new QDataStream(&this->client());

    // connect signals
    connect(&this->client(), SIGNAL(readyRead()), this, SLOT(readyRead()));
    connect(&this->client(), SIGNAL(disconnected()), this, SLOT(disconnected()));

    // close server
    this->close();
}

void VoltageServer::readyRead() {
    // read matrix
    if (this->voltage_ != NULL) {
        for (fastEIT::dtype::index column = 0; column < this->voltage().columns(); ++column) {
            for (fastEIT::dtype::index row = 0; row < this->voltage().rows(); ++row) {
                this->input_stream() >> this->voltage()(row, column);
            }
        }

        // upload to device
        this->voltage().copyToDevice(NULL);
    }

    // clear buffer
    this->client().readAll();
}

void VoltageServer::disconnected() {
    // clear member
    this->client_ = NULL;
    this->input_stream_ = NULL;

    // open server
    this->listen(QHostAddress::Any, 3000);
}
