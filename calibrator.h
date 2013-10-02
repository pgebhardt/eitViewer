#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "solver.h"
#include "firfilter.h"
#include <QTimer>

class Calibrator : public Solver {
    Q_OBJECT
public:
    explicit Calibrator(Solver* differential_solver, const QJsonObject& config,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
        int cuda_device=0, QObject* parent=nullptr);
    virtual ~Calibrator();

public slots:
    void init_filter(int order, int step_size);
    void solve();
    void start(int step_size);
    void stop();

public:
    // accessor
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> image() {
        return this->eit_solver()->gamma();
    }
    Solver* differential_solver() { return this->differential_solver_; }
    FIRFilter* filter() { return this->filter_; }
    QTimer* timer() { return this->timer_; }
    bool running() { return this->timer()->isActive(); }
    int& step_size() { return this->step_size_; }

private:
    // member
    Solver* differential_solver_;
    FIRFilter* filter_;
    QTimer* timer_;
    int step_size_;
};

#endif // CALIBRATOR_H
