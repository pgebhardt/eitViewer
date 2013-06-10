#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    int cuda_device, QObject *parent)
    : Solver(config, cuda_device, parent), differential_solver_(differential_solver),
    filter_(nullptr), running_(false) {
    connect(this, &Calibrator::initialized, [=](bool success) {
        if (success) {
            // set regularization factor
            this->fasteit_solver()->inverse_solver()->regularization_factor() =
                config["calibrator"].toObject()["regularization_factor"].toDouble();

            // reset timer
            this->restart((int)config["calibrator"].toObject()["step_size"].toDouble());
        }
    });

    connect(this->differential_solver(), &Solver::initialized, [=](bool success) {
        if (success) {
            QMetaObject::invokeMethod(this, "init_filter", Qt::AutoConnection,
                Q_ARG(int, (int)config["calibrator"].toObject()["filter_order"].toDouble()),
                Q_ARG(int, (int)config["calibrator"].toObject()["filter_step_size"].toDouble()));
        }
    });
}

Calibrator::~Calibrator() {
    // cleanup filter
    this->filter()->thread()->quit();
    this->filter()->thread()->wait();
    delete this->filter_;
}

void Calibrator::init_filter(int order, int step_size) {
    this->filter_ = new FIRFilter(order, step_size, this->cuda_device(),
        this->differential_solver()->measurement());
}

void Calibrator::solve() {
    if (this->running()) {
        this->time().restart();

        cudaStreamSynchronize(this->filter()->cuda_stream());
        this->measurement()->copy(this->filter()->output(),
            this->cuda_stream());

        this->fasteit_solver()->solve_absolute(this->cublas_handle(),
            this->cuda_stream())->copyToHost(this->cuda_stream());

        this->differential_solver()->calculation()->copy(
            this->fasteit_solver()->forward_solver()->voltage(),
            this->cuda_stream());

        this->solve_time() = this->time().elapsed();
    }
}
