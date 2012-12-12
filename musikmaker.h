#ifndef MUSIKMAKER_H
#define MUSIKMAKER_H

#include <QObject>
#include <QSound>
#include <QMap>
#include <fasteit/fasteit.h>

class MusikMaker : public QObject
{
    Q_OBJECT
public:
    explicit MusikMaker(std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model,
                        QObject *parent = 0);

    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> getPosition(
        std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> gamma,
        fastEIT::dtype::real threshold);
    QString getNode(std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> position);
    void playNode(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> gamma,
                  fastEIT::dtype::real threshold);


signals:
    
public slots:

public:
    // accessors
    const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() const {
        return this->model_;
    }
    const QMap<QString, QSound*>& sounds() const { return this->sounds_; }
    const QString& previous_node() const { return this->previous_node_; }

    // mutator
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() {
        return this->model_;
    }
    QMap<QString, QSound*>& sounds() { return this->sounds_; }
    QString& previous_node() { return this->previous_node_; }

private:
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model_;
    QMap<QString, QSound*> sounds_;
    QString previous_node_;
};

#endif // MUSIKMAKER_H
