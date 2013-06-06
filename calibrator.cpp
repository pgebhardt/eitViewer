#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    int cuda_device, QObject *parent)
    : Solver(config, cuda_device, parent), differential_solver_(differential_solver),
    filter_(nullptr), running_(false), cuda_device_(cuda_device) {
    connect(this, &Calibrator::initialized, [=](bool success) {
        if (success) {
            // set regularization factor
            this->fasteit_solver()->inverse_solver()->regularization_factor() =
                config["calibrator"].toObject()["regularization_factor"].toDouble();

            // reset timer
            this->restart((int)config["calibrator"].toObject()["step_size"].toDouble());
        }
    });

    connect(this->differential_solver(), &Solver::initialized, this, &Calibrator::init_filter);
}

Calibrator::~Calibrator() {
    // cleanup filter
    this->filter()->thread()->quit();
    this->filter()->thread()->wait();
    delete this->filter_;
}

void Calibrator::init_filter(bool success) {
    if (success) {
        this->filter_ = new FIRFilter(10, 100, this->cuda_device(),
            this->differential_solver()->measured_voltage());
    }
}

void Calibrator::solve() {
    if (this->running()) {
        this->time().restart();

        cudaStreamSynchronize(this->filter()->cuda_stream());
        this->measured_voltage()->copy(this->filter()->output(),
            this->cuda_stream());

        this->fasteit_solver()->solve_absolute(this->cublas_handle(),
            this->cuda_stream())->copyToHost(this->cuda_stream());

        this->differential_solver()->calculated_voltage()->copy(
            this->fasteit_solver()->forward_solver()->voltage(),
            this->cuda_stream());

        this->solve_time() = this->time().elapsed();
    }
}
