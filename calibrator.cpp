#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
    int cuda_device, QObject *parent)
    : Solver(config, nodes, elements, boundary, 1, cuda_device, parent),
    differential_solver_(differential_solver), filter_(nullptr), running_(false),
    step_size_(20) {
    connect(this, &Calibrator::initialized, [=](bool success) {
        if (success) {
            // set regularization factor
            this->eit_solver()->inverse_solver()->regularization_factor() =
                config["calibrator"].toObject()["regularization_factor"].toDouble();

            // create and start timer
            this->timer_ = new QTimer(this);
            connect(this->timer(), &QTimer::timeout, this, &Calibrator::solve);
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

void Calibrator::restart(int step_size) {
    this->step_size_ = step_size;
    this->timer()->start(this->step_size());
}

void Calibrator::init_filter(int order, int step_size) {
    this->filter_ = new FIRFilter(order, step_size, this->cuda_device(),
        this->differential_solver()->eit_solver()->measurement()[0]);
}

void Calibrator::solve() {
    if (this->running()) {
        this->time().restart();

        cudaStreamSynchronize(this->filter()->cuda_stream());
        this->eit_solver()->measurement()[0]->copy(this->filter()->output(),
            this->cuda_stream());

        this->eit_solver()->solve_absolute(this->cublas_handle(),
            this->cuda_stream())->copyToHost(this->cuda_stream());

        this->differential_solver()->eit_solver()->calculation()[0]->copy(
            this->eit_solver()->forward_solver()->voltage(),
            this->cuda_stream());

        this->solve_time() = this->time().elapsed();
    }
}
