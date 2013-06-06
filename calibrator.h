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
    void init_filter(int order, int step_size);

public:
    // accessor
    Solver* differential_solver() { return this->differential_solver_; }
    FIRFilter* filter() { return this->filter_; }
    bool& running() { return this->running_; }

private:
    // member
    Solver* differential_solver_;
    FIRFilter* filter_;
    bool running_;
};

#endif // CALIBRATOR_H
