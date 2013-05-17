#include "solver.h"

template <
    class type
>
std::shared_ptr<fastEIT::Matrix<type>> matrixFromJsonArray(const QJsonArray& array, cudaStream_t stream) {
    auto matrix = std::make_shared<fastEIT::Matrix<type>>(array.size(), array.first().toArray().size(),
        stream);
    for (fastEIT::dtype::index row = 0; row < matrix->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < matrix->columns(); ++column) {
        (*matrix)(row, column) = array[row].toArray()[column].toDouble();
    }
    matrix->copyToDevice(nullptr);

    return matrix;
}

Solver::Solver(const QJsonObject& config, QObject *parent) :
    QObject(parent) {
    // create cublas handle
    cublasCreate(&this->handle_);

    // init separate thread
    this->thread_ = new QThread();
    this->moveToThread(this->thread());

    // create solver once thread is started
    connect(this->thread(), &QThread::started,
        [=] () {
            std::cout << "solver thread: " << QThread::currentThreadId() << std::endl;

            // load mesh from config
            auto nodes = matrixFromJsonArray<fastEIT::dtype::real>(
                config["model"].toObject()["mesh"].toObject()["nodes"].toArray(), nullptr);
            auto elements = matrixFromJsonArray<fastEIT::dtype::index>(
                config["model"].toObject()["mesh"].toObject()["elements"].toArray(), nullptr);
            auto boundary = matrixFromJsonArray<fastEIT::dtype::index>(
                config["model"].toObject()["mesh"].toObject()["boundary"].toArray(), nullptr);

            // load pattern from config
            auto drive_pattern = matrixFromJsonArray<fastEIT::dtype::real>(
                config["model"].toObject()["source"].toObject()["drive_pattern"].toArray(), nullptr);
            auto measurement_pattern = matrixFromJsonArray<fastEIT::dtype::real>(
                config["model"].toObject()["source"].toObject()["measurement_pattern"].toArray(), nullptr);

            // create mesh
            auto mesh = fastEIT::mesh::quadraticBasis(nodes, elements, boundary,
                config["model"].toObject()["mesh"].toObject()["radius"].toDouble(),
                config["model"].toObject()["mesh"].toObject()["height"].toDouble(),
                nullptr);

            // create electrodes
            auto electrodes = fastEIT::electrodes::circularBoundary(
                config["model"].toObject()["electrodes"].toObject()["count"].toDouble(),
                std::make_tuple(config["model"].toObject()["electrodes"].toObject()["width"].toDouble(),
                    config["model"].toObject()["electrodes"].toObject()["height"].toDouble()),
                1.0, mesh->radius());

            // create source
            auto source = std::make_shared<fastEIT::source::Current<fastEIT::basis::Quadratic>>(
                config["model"].toObject()["source"].toObject()["current"].toDouble(),
                mesh, electrodes, config["model"].toObject()["components_count"].toDouble(),
                drive_pattern, measurement_pattern, this->handle(), nullptr);

            // create model
            auto model = std::make_shared<fastEIT::Model<fastEIT::basis::Quadratic>>(
                mesh, electrodes, source, config["model"].toObject()["sigma_ref"].toDouble(),
                config["model"].toObject()["components_count"].toDouble(), this->handle(),
                nullptr);

            // create and init solver
            this->fasteit_solver_ = std::make_shared<fastEIT::Solver>(model,
                config["solver"].toObject()["regularization_factor"].toDouble(),
                this->handle(), nullptr);
            this->fasteit_solver()->preSolve(this->handle(), nullptr);
            this->fasteit_solver()->measured_voltage()->copyToHost(nullptr);

            // signal solver ready
            emit this->initialized();

            // start solve timer
            this->timer_ = new QTimer();
            connect(this->timer(), &QTimer::timeout, [=]() {
                this->time().restart();

                auto gamma = this->fasteit_solver()->solve(this->handle(), nullptr);
                gamma->copyToHost(nullptr);

                this->solve_time() = this->time().elapsed();
            });
            this->timer()->start(10);
    });

    // start thread
    this->thread()->start();
}
