#include "mirrorserver.h"
#include <QDataStream>
#include <QJsonObject>
#include <QJsonDocument>

MirrorServer::MirrorServer(Image* image, std::vector<std::tuple<QString, QString>>* analysis, QObject* parent) :
    _image(image), _analysis(analysis), QObject(parent) {
    // create http server
    this->_httpServer = new QHttpServer(this);
    connect(this->httpServer(), &QHttpServer::newRequest, this, &MirrorServer::handleRequest);

    // start listening
    this->httpServer()->listen(QHostAddress::Any, 3003);
}

void MirrorServer::handleRequest(QHttpRequest *request, QHttpResponse *response) {
    Q_UNUSED(response);

    // call response handler according to request type
    if (request->path() == "/electrodes") {
        this->handleElectrodesConfigRequest(response);
    }
    else if (request->path() == "/vertices") {
        this->handleVerticesConfigRequest(response);
    }
    else if (request->path() == "/colors") {
        this->handleColorConfigRequest(response);
    }
    else if (request->path() == "/vertices-update") {
        this->handleVerticesUpdateRequest(response);
    }
    else if (request->path() == "/colors-update") {
        this->handleColorUpdateRequest(response);
    }
    else if (request->path() == "/analysis-update") {
        this->handleAnalysisUpdateRequest(response);
    }
    else if (request->path() == "/calibrate") {
        this->handleCalibrateRequest(response);
    }
}

void MirrorServer::handleElectrodesConfigRequest(QHttpResponse *response) {
    QVariantMap map;
    map["count"] = (int)this->image()->electrodes().cols();
    map["length"] = 0.1;

    QJsonObject json = QJsonObject::fromVariantMap(map);
    QJsonDocument jsonDocument(json);
    QByteArray body = jsonDocument.toJson();

    response->setHeader("Content-Length", QString::number(body.length()));
    response->writeHead(200);
    response->end(body);
}

void MirrorServer::handleVerticesConfigRequest(QHttpResponse *response) {
    // fill dictionary with config data
    QByteArray vertices = QByteArray::fromRawData((const char*)this->image()->vertices().data(),
        sizeof(float) * this->image()->vertices().rows() * this->image()->vertices().cols());

    response->setHeader("Content-Length", QString::number(vertices.length()));
    response->writeHead(200);
    response->end(vertices);
}

void MirrorServer::handleVerticesUpdateRequest(QHttpResponse *response) {
    // fill dictionary with config data
    Eigen::ArrayXXf vertices = Eigen::ArrayXXf::Zero(3, this->image()->vertices().cols());
    vertices.row(0) = this->image()->vertices().row(2 + 0 * 3).eval();
    vertices.row(1) = this->image()->vertices().row(2 + 1 * 3).eval();
    vertices.row(2) = this->image()->vertices().row(2 + 2 * 3).eval();
    QByteArray verticesByteArray = QByteArray::fromRawData((const char*)vertices.data(),
        sizeof(float) * vertices.rows() * vertices.cols());

    response->setHeader("Content-Length", QString::number(verticesByteArray.length()));
    response->writeHead(200);
    response->end(verticesByteArray);
}

void MirrorServer::handleColorConfigRequest(QHttpResponse *response) {
    // fill dictionary with config data
    QByteArray colors = QByteArray::fromRawData((const char*)this->image()->colors().data(),
        sizeof(float) * this->image()->colors().rows() * this->image()->colors().cols());

    response->setHeader("Content-Length", QString::number(colors.length()));
    response->writeHead(200);
    response->end(colors);
}

void MirrorServer::handleColorUpdateRequest(QHttpResponse* response){
    // fill dictionary with config data
    Eigen::ArrayXXf colors = this->image()->colors().block(0, 0, 3, this->image()->colors().cols()).eval();
    QByteArray colorsByteArray = QByteArray::fromRawData((const char*)colors.data(),
        sizeof(float) * colors.rows() * colors.cols());

    response->setHeader("Content-Length", QString::number(colorsByteArray.length()));
    response->writeHead(200);
    response->end(colorsByteArray);
}

void MirrorServer::handleAnalysisUpdateRequest(QHttpResponse *response) {
    QVariantList list;
    for (const auto& analysis : this->analysis()) {
        QVariantMap analysisMap;
        analysisMap["name"] = std::get<0>(analysis);
        analysisMap["result"] = std::get<1>(analysis);
        list.append(analysisMap);
    }

    QVariantMap map;
    map["analysis"] = list;

    QJsonObject json = QJsonObject::fromVariantMap(map);
    QJsonDocument jsonDocument(json);
    QByteArray body = jsonDocument.toJson();

    response->setHeader("Content-Length", QString::number(body.length()));
    response->writeHead(200);
    response->end(body);
}

void MirrorServer::handleCalibrateRequest(QHttpResponse *response) {
    Q_UNUSED(response);

    emit this->calibrate();
}
