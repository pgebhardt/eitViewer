#include "solver.h"

template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> matrixFromJsonArray(const QJsonArray& array, cudaStream_t cuda_stream) {
    auto matrix = std::make_shared<mpFlow::numeric::Matrix<type>>(array.size(), array.first().toArray().size(),
        cuda_stream);
    for (mpFlow::dtype::index row = 0; row < matrix->rows(); ++row)
    for (mpFlow::dtype::index column = 0; column < matrix->columns(); ++column) {
        (*matrix)(row, column) = array[row].toArray()[column].toDouble();
    }
    matrix->copyToDevice(cuda_stream);

    return matrix;
}

Solver::Solver(const QJsonObject& config, int cuda_device, QObject *parent) :
    QObject(parent), cuda_stream_(nullptr), cublas_handle_(nullptr), cuda_device_(cuda_device),
    step_size_(20) {
    // init separate thread
    this->thread_ = new QThread(this);
    this->moveToThread(this->thread());

    // create solver once thread is started
    connect(this->thread(), &QThread::started,
        [=] () {
        // switch to cuda device and init cublas
        cudaSetDevice(this->cuda_device());
        cublasCreate(&this->cublas_handle_);

        bool success = true;
        try {
            // load mesh from config
            auto nodes = matrixFromJsonArray<mpFlow::dtype::real>(
                config["model"].toObject()["mesh"].toObject()["nodes"].toArray(), this->cuda_stream());
            auto elements = matrixFromJsonArray<mpFlow::dtype::index>(
                config["model"].toObject()["mesh"].toObject()["elements"].toArray(), this->cuda_stream());
            auto boundary = matrixFromJsonArray<mpFlow::dtype::index>(
                config["model"].toObject()["mesh"].toObject()["boundary"].toArray(), this->cuda_stream());

            // load pattern from config
            auto drive_pattern = matrixFromJsonArray<mpFlow::dtype::real>(
                config["model"].toObject()["source"].toObject()["drive_pattern"].toArray(), this->cuda_stream());
            auto measurement_pattern = matrixFromJsonArray<mpFlow::dtype::real>(
                config["model"].toObject()["source"].toObject()["measurement_pattern"].toArray(), this->cuda_stream());

            // create mesh
            auto mesh = mpFlow::numeric::irregularMesh::quadraticBasis(nodes, elements, boundary,
                config["model"].toObject()["mesh"].toObject()["radius"].toDouble(),
                config["model"].toObject()["mesh"].toObject()["height"].toDouble(),
                this->cuda_stream());

            // create electrodes
            auto electrodes = mpFlow::EIT::electrodes::circularBoundary(
                config["model"].toObject()["electrodes"].toObject()["count"].toDouble(),
                std::make_tuple(config["model"].toObject()["electrodes"].toObject()["width"].toDouble(),
                    config["model"].toObject()["electrodes"].toObject()["height"].toDouble()),
                1.0, mesh->radius());

            // create source
            auto source = std::make_shared<mpFlow::EIT::source::Current<mpFlow::EIT::basis::Quadratic>>(
                config["model"].toObject()["source"].toObject()["current"].toDouble(),
                mesh, electrodes, config["model"].toObject()["components_count"].toDouble(),
                drive_pattern, measurement_pattern, this->cublas_handle(), this->cuda_stream());

            // create model
            auto model = std::make_shared<mpFlow::EIT::Model<mpFlow::EIT::basis::Quadratic>>(
                mesh, electrodes, source, config["model"].toObject()["sigma_ref"].toDouble(),
                config["model"].toObject()["components_count"].toDouble(), this->cublas_handle(),
                this->cuda_stream());

            // create and init solver
            this->eit_solver_ = std::make_shared<mpFlow::EIT::solver::Solver<
                    mpFlow::numeric::SparseConjugate, mpFlow::numeric::FastConjugate>>(model, 1,
                config["solver"].toObject()["regularization_factor"].toDouble(),
                this->cublas_handle(), this->cuda_stream());
            this->eit_solver()->preSolve(this->cublas_handle(), this->cuda_stream());
            this->measurement()->copyToHost(this->cuda_stream());

            // start solve timer
            this->solve_timer_ = new QTimer(this);
            connect(this->solve_timer(), &QTimer::timeout, this, &Solver::solve);
            this->restart(20);

        } catch (const std::exception& e) {
            success = false;
        }

        // signal solver ready
        emit this->initialized(success);
    });

    // start thread
    this->thread()->start();
}

void Solver::restart(int step_size) {
    this->step_size_ = step_size;
    this->solve_timer()->start(this->step_size());
}

void Solver::solve() {
    this->time().restart();

    this->eit_solver()->solve_differential(this->cublas_handle(),
        this->cuda_stream())->copyToHost(this->cuda_stream());

    this->solve_time() = this->time().elapsed();
}
