#ifndef MUSIKMAKER_H
#define MUSIKMAKER_H

#include <QObject>
#include <QSound>
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

    std::string getNode(std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> position);

signals:
    
public slots:

public:
    // accessors
    const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() const {
        return this->model_;
    }
    const QSound* sound(int index) const { return this->sounds_[index]; }
    const std::string& previous_node() const { return this->previous_node_; }

    // mutator
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() {
        return this->model_;
    }
    QSound* sound(int index) { return this->sounds_[index]; }
    std::string& previous_node() { return this->previous_node_; }

private:
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model_;
    std::vector<QSound*> sounds_;
    std::string previous_node_;
};

#endif // MUSIKMAKER_H
