#ifndef MIRRORSERVER_H
#define MIRRORSERVER_H

#include <QObject>
#include <qhttpserver.h>
#include <qhttprequest.h>
#include <qhttpresponse.h>
#include "image.h"

class MirrorServer : public QObject {
    Q_OBJECT
public:
    explicit MirrorServer(Image* image, QObject *parent = 0);

public:
    void handleRequest(QHttpRequest* request, QHttpResponse* response);

protected:
    void handleElectrodesConfigRequest(QHttpResponse* response);
    void handleVerticesConfigRequest(QHttpResponse* response);
    void handleVerticesUpdateRequest(QHttpResponse* response);
    void handleColorConfigRequest(QHttpResponse* response);
    void handleColorUpdateRequest(QHttpResponse* response);

public:
    QHttpServer* httpServer() { return this->_httpServer; }
    Image* image() { return this->_image; }

private:
    QHttpServer* _httpServer;
    Image* _image;
};

#endif // MIRRORSERVER_H
