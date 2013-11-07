#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
    int cuda_device, QObject *parent)
    : Solver(config, nodes, elements, boundary, 1, cuda_device, parent),
    differential_solver_(differential_solver), step_size_(20),
    buffer_pos_(0.0), buffer_increment_(0.0) {
    connect(this, &Calibrator::initialized, [=](bool success) {
        if (success) {
            // set regularization factor
            this->eit_solver()->inverse_solver()->regularization_factor() =
                config["calibrator"].toObject()["regularization_factor"].toDouble();

            // create and start timer
            this->timer_ = new QTimer(this);
            connect(&this->timer(), &QTimer::timeout, this, &Calibrator::solve);
            this->step_size() = (int)config["calibrator"].toObject()["step_size"].toDouble();
        }
    });
}

void Calibrator::update_data(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>> *data,
    double time_elapsed) {
    this->data() = *data;

    // update buffer increment
    this->buffer_pos() = 0.0;
    this->buffer_increment() = time_elapsed > ((double)this->step_size() * 1e-3) ?
        ((double)this->step_size() * 1e-3 / time_elapsed * (double)data->size()) :
        0.0;

    // start calibrator timer
    this->timer().start(this->step_size());
}

void Calibrator::solve() {
    this->time().restart();

    // copy current data set to solver
    this->eit_solver()->measurement()[0]->copy(
        this->data()[this->buffer_pos()], this->cuda_stream());

    this->eit_solver()->solve_absolute(this->cublas_handle(),
        this->cuda_stream())->copyToHost(this->cuda_stream());

    for (mpFlow::dtype::index i = 0; i < this->differential_solver()->eit_solver()->calculation().size(); ++i) {
        this->differential_solver()->eit_solver()->calculation()[i]->copy(
            this->eit_solver()->forward_solver()->voltage(),
            this->cuda_stream());
    }

    // update buffer pos
    this->buffer_pos() += this->buffer_increment();
    if (this->buffer_pos() >= (double)this->data().size()) {
        this->buffer_pos() = (double)this->data().size() - 1.0;

        // stop timer
        this->timer().stop();
    }

    cudaStreamSynchronize(this->cuda_stream());
    this->solve_time() = this->time().elapsed();
}
