#ifndef MUSIKMAKER_H
#define MUSIKMAKER_H

#include <QObject>
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

    // mutator
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() {
        return this->model_;
    }

private:
    std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model_;
};

#endif // MUSIKMAKER_H
