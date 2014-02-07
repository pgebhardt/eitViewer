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
    explicit MirrorServer(Image* image, std::vector<std::tuple<QString, QString>>* analysis,
                          QObject *parent = 0);

signals:
    void calibrate();

public:
    void handleRequest(QHttpRequest* request, QHttpResponse* response);

protected:
    void handleElectrodesConfigRequest(QHttpResponse* response);
    void handleVerticesConfigRequest(QHttpResponse* response);
    void handleVerticesUpdateRequest(QHttpResponse* response);
    void handleColorConfigRequest(QHttpResponse* response);
    void handleColorUpdateRequest(QHttpResponse* response);
    void handleAnalysisUpdateRequest(QHttpResponse* response);
    void handleCalibrateRequest(QHttpResponse* response);

public:
    QHttpServer* httpServer() { return this->_httpServer; }
    Image* image() { return this->_image; }
    std::vector<std::tuple<QString, QString>>& analysis() { return *this->_analysis; }

private:
    QHttpServer* _httpServer;
    Image* _image;
    std::vector<std::tuple<QString, QString>>* _analysis;
};

#endif // MIRRORSERVER_H
