#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    int cuda_device, QObject *parent)
    : Solver(config, cuda_device, parent), differential_solver_(differential_solver),
    running_(false) {
    connect(this, &Calibrator::initialized, [=](bool) {
        // start timer
        this->solve_timer()->start(500);

        // set regularization factor
        this->fasteit_solver()->inverse_solver()->regularization_factor() = 1e5;
    });
}

void Calibrator::solve() {
    if (this->running()) {
        this->time().restart();

        this->measured_voltage()->copy(this->differential_solver()->measured_voltage(),
            this->cuda_stream());
        this->fasteit_solver()->solve_absolute(this->cublas_handle(), this->cuda_stream());
        this->differential_solver()->calculated_voltage()->copy(
            this->fasteit_solver()->forward_solver()->voltage(),
            this->cuda_stream());

        this->solve_time() = this->time().elapsed();
    }
}
