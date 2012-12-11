#ifndef VOLTAGESERVER_H
#define VOLTAGESERVER_H

#include <QTcpSocket>
#include <QTcpServer>
#include <QThread>
#include <fasteit/fasteit.h>

class VoltageServer : public QTcpServer
{
    Q_OBJECT
public:
    explicit VoltageServer(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> voltage,
                           QObject *parent = 0);

signals:
    
public slots:
    virtual void init();
    virtual void acceptConnection();
    virtual void readyRead();
    virtual void diconnected();

public:
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> voltage() const {
        return this->voltage_;
    }
    const QTcpSocket* client() const { return this->client_; }
    const QThread* thread() const { return this->thread_; }

    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> voltage() {
        return this->voltage_;
    }
    QTcpSocket* client() { return this->client_; }
    QThread* thread() { return this->thread_; }

private:
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> voltage_;
    QTcpSocket* client_;
    QThread* thread_;
};

#endif // VOLTAGESERVER_H
