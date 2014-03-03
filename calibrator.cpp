#include "calibrator.h"

Calibrator::Calibrator(Solver* differential_solver, const QJsonObject& config,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
    int cuda_device, QObject *parent)
    : Solver(config, nodes, elements, boundary, 1, cuda_device, parent),
    differential_solver_(differential_solver), filteredData_(nullptr),
    offset_(nullptr), step_size_(2000), filterConstant_(10.0) {
    connect(this, &Calibrator::initialized, [=](bool success) {
        if (success) {
            // set regularization factor
            this->eit_solver()->inverse_solver()->regularization_factor() =
                config["calibrator"].toObject()["regularization_factor"].toDouble();

            // create and start timer
            this->timer_ = new QTimer(this);
            connect(&this->timer(), &QTimer::timeout, this, &Calibrator::solve);
            this->step_size() = (int)(1e3 * config["calibrator"].toObject()["calibration_interval"].toDouble());

            // set filter constant
            this->filterConstant() = config["calibrator"].toObject()["filter_constant"].toDouble();
        }
    });
}

void Calibrator::stop() {
    this->timer().stop();
    this->offset_ = nullptr;
    this->filteredData_ = nullptr;
}

void Calibrator::update_data(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>> *data,
    double time_elapsed) {
    // create filtered data matrix, if neccessary
    if (this->filteredData() == nullptr) {
        this->filteredData_ = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::real>>(
            (*data)[0]->rows(), (*data)[0]->columns(), this->cuda_stream());
        this->filteredData()->copy((*data)[0], this->cuda_stream());
    }

    // perform IIR low pass filter function
    mpFlow::dtype::real deltaT = time_elapsed / (mpFlow::dtype::real)data->size();
    mpFlow::dtype::real alpha = deltaT / (this->filterConstant() + deltaT);
    for (const auto& measurementData : *data) {
        this->filteredData()->scalarMultiply(1.0 - alpha, this->cuda_stream());
        cublasSaxpy(this->cublas_handle(), this->filteredData()->data_rows() * this->filteredData()->data_columns(),
            &alpha, measurementData->device_data(), 1, this->filteredData()->device_data(), 1);
    }

    // set offest matrix if not already set
    if (this->offset() == nullptr) {
        this->offset_ = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::real>>(
            this->eit_solver()->measurement()[0]->rows(), this->eit_solver()->measurement()[0]->columns(),
            this->cuda_stream());
        this->offset()->copy(this->filteredData(), this->cuda_stream());
        this->offset()->scalarMultiply(-1.0, this->cuda_stream());
        this->offset()->add(this->eit_solver()->forward_solver()->voltage(), this->cuda_stream());
    }

    // start calibrator timer
    if (!this->timer().isActive()) {
        this->timer().start(this->step_size());
    }
}

void Calibrator::solve() {
    this->time().restart();

    // copy current data set to solver and add offset
    this->eit_solver()->measurement()[0]->copy(this->filteredData(), this->cuda_stream());
    this->eit_solver()->measurement()[0]->add(this->offset(), this->cuda_stream());

    this->eit_solver()->solve_absolute(this->cublas_handle(),
        this->cuda_stream())->copyToHost(this->cuda_stream());

    for (mpFlow::dtype::index i = 0; i < this->differential_solver()->eit_solver()->calculation().size(); ++i) {
        this->differential_solver()->eit_solver()->calculation()[i]->copy(this->offset(), this->cuda_stream());
        this->differential_solver()->eit_solver()->calculation()[i]->scalarMultiply(-1.0, this->cuda_stream());
        this->differential_solver()->eit_solver()->calculation()[i]->add(
            this->eit_solver()->forward_solver()->voltage(), this->cuda_stream());
    }

    cudaStreamSynchronize(this->cuda_stream());
    this->solve_time() = this->time().elapsed();
}
