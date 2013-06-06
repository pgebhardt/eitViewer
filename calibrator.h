#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "solver.h"
#include "firfilter.h"

class Calibrator : public Solver {
    Q_OBJECT
public:
    explicit Calibrator(Solver* differential_solver, const QJsonObject& config,
        int cuda_device=0, QObject* parent=nullptr);
    virtual ~Calibrator();

protected slots:
    virtual void solve();
    void init_filter(bool success);

public:
    // accessor
    Solver* differential_solver() { return this->differential_solver_; }
    FIRFilter* filter() { return this->filter_; }
    bool& running() { return this->running_; }
    int cuda_device() { return this->cuda_device_; }

private:
    // member
    Solver* differential_solver_;
    FIRFilter* filter_;
    bool running_;
    int cuda_device_;
};

#endif // CALIBRATOR_H
