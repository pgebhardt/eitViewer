#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "solver.h"
#include <QTimer>

class Calibrator : public Solver {
    Q_OBJECT
public:
    explicit Calibrator(Solver* differential_solver, const QJsonObject& config,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
        int cuda_device=0, QObject* parent=nullptr);
    void stop();

public slots:
    void update_data(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* data,
        double time_elapsed);
    void solve();

public:
    // accessor
    Solver* differential_solver() { return this->differential_solver_; }
    QTimer& timer() { return *this->timer_; }
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>& data() {
        return this->data_;
    }
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> offset() { return this->offset_; }
    int& step_size() { return this->step_size_; }
    double& buffer_pos() { return this->buffer_pos_; }
    double& buffer_increment() { return this->buffer_increment_; }

private:
    // member
    Solver* differential_solver_;
    QTimer* timer_;
    std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>> data_;
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> offset_;
    int step_size_;
    double buffer_pos_;
    double buffer_increment_;
};

#endif // CALIBRATOR_H
