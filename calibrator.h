#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "solver.h"

class Calibrator : public Solver {
    Q_OBJECT
public:
    explicit Calibrator(Solver* differential_solver, const QJsonObject& config,
        int cuda_device=0, QObject* parent=nullptr);
    
protected slots:
    virtual void solve();

public:
    // accessor
    Solver* differential_solver() { return this->differential_solver_; }

private:
    // member
    Solver* differential_solver_;
};

#endif // CALIBRATOR_H
