#ifndef VOLTAGESERVER_H
#define VOLTAGESERVER_H

#include <QTcpServer>
#include <QTcpSocket>
#include <QThread>
#include <QDataStream>
#include <fasteit/fasteit.h>

class VoltageServer : public QTcpServer
{
    Q_OBJECT
public:
    explicit VoltageServer(fastEIT::Matrix<fastEIT::dtype::real>* voltage,
                           QObject *parent = 0);
    
signals:
    
public slots:
    virtual void acceptConnection();
    virtual void readyRead();
    virtual void disconnected();
    virtual void init();

public:
    // accessor
    const fastEIT::Matrix<fastEIT::dtype::real>& voltage() const {
        return *this->voltage_;
    }
    const QThread& thread() const { return *this->thread_; }
    const QTcpSocket& client() const { return *this->client_; }
    const QDataStream& input_stream() const { return *this->input_stream_; }

    // mutators
    fastEIT::Matrix<fastEIT::dtype::real>& voltage() { return *this->voltage_; }
    QThread& thread() { return *this->thread_; }
    QTcpSocket& client() { return *this->client_; }
    QDataStream& input_stream() { return *this->input_stream_; }

// member
private:
    fastEIT::Matrix<fastEIT::dtype::real>* voltage_;
    QThread* thread_;
    QTcpSocket* client_;
    QDataStream* input_stream_;
};

#endif // VOLTAGESERVER_H
