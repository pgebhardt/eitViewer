#include "solver.h"
#include <distmesh/distmesh.h>

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

std::shared_ptr<mpFlow::EIT::solver::Solver<mpFlow::numeric::Conjugate>>
    Solver::createSolverFromConfig(const QJsonObject &config,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
    int parallel_images, cublasHandle_t handle, cudaStream_t stream) {
    // load pattern from config
    auto drive_pattern = matrixFromJsonArray<mpFlow::dtype::real>(
        config["model"].toObject()["source"].toObject()["drive_pattern"].toArray(), stream);
    auto measurement_pattern = matrixFromJsonArray<mpFlow::dtype::real>(
        config["model"].toObject()["source"].toObject()["measurement_pattern"].toArray(), stream);

    // read out type of basis function
    auto basis_function_type = config["model"].toObject()["basis_function"].toString();

    // create mesh
    std::shared_ptr<mpFlow::numeric::IrregularMesh> mesh = nullptr;
    if (basis_function_type == "quadratic") {
        mesh = mpFlow::numeric::irregularMesh::quadraticBasis(nodes, elements, boundary,
            config["model"].toObject()["mesh"].toObject()["radius"].toDouble(),
            config["model"].toObject()["mesh"].toObject()["height"].toDouble(),
            stream);
    } else {
        mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(nodes, elements, boundary,
            config["model"].toObject()["mesh"].toObject()["radius"].toDouble(),
            config["model"].toObject()["mesh"].toObject()["height"].toDouble());
    }

    // create electrodes
    auto electrodes = mpFlow::EIT::electrodes::circularBoundary(
        config["model"].toObject()["electrodes"].toObject()["count"].toDouble(),
        std::make_tuple(config["model"].toObject()["electrodes"].toObject()["width"].toDouble(),
            config["model"].toObject()["electrodes"].toObject()["height"].toDouble()),
        1.0, mesh->radius());

    // read out current
    std::vector<mpFlow::dtype::real> current(drive_pattern->columns());
    if (config["model"].toObject()["source"].toObject()["current"].isArray()) {
        for (mpFlow::dtype::index i = 0; i < drive_pattern->columns(); ++i) {
            current[i] = config["model"].toObject()["source"].toObject()["current"].toArray()[i].toDouble();
        }
    } else {
        std::fill(current.begin(), current.end(),
            config["model"].toObject()["source"].toObject()["current"].toDouble());
    }

    // create source
    std::shared_ptr<mpFlow::EIT::source::Source> source = nullptr;
    if (basis_function_type == "quadratic") {
        source = std::make_shared<mpFlow::EIT::source::Current<mpFlow::FEM::basis::Quadratic>>(
            current, mesh, electrodes, config["model"].toObject()["components_count"].toDouble(),
            drive_pattern, measurement_pattern, handle, stream);
    } else {
        source = std::make_shared<mpFlow::EIT::source::Current<mpFlow::FEM::basis::Linear>>(
            current, mesh, electrodes, config["model"].toObject()["components_count"].toDouble(),
            drive_pattern, measurement_pattern, handle, stream);
    }

    // create model
    std::shared_ptr<mpFlow::EIT::model::Base> model = nullptr;
    if (basis_function_type == "quadratic") {
        model = std::make_shared<mpFlow::EIT::Model<mpFlow::FEM::basis::Quadratic>>(
            mesh, electrodes, source, config["model"].toObject()["sigma_ref"].toDouble(),
            config["model"].toObject()["components_count"].toDouble(), handle, stream);
    } else {
        model = std::make_shared<mpFlow::EIT::Model<mpFlow::FEM::basis::Linear>>(
            mesh, electrodes, source, config["model"].toObject()["sigma_ref"].toDouble(),
            config["model"].toObject()["components_count"].toDouble(), handle, stream);
    }

    // create and init solver
    return std::make_shared<mpFlow::EIT::solver::Solver<mpFlow::numeric::Conjugate>>(
        model, parallel_images, config["solver"].toObject()["regularization_factor"].toDouble(),
        handle, stream);
}

std::tuple<
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>>
    Solver::createMeshFromConfig(const QJsonObject& config, cudaStream_t stream) {
    // extract parameter from config
    distmesh::dtype::real radius = config["radius"].toDouble();
    distmesh::dtype::array<distmesh::dtype::real> bounding_box(2, 2);
    bounding_box << -1.1 * radius, 1.1 * radius, -1.1 * radius, 1.1 * radius;

    // create mesh using libdistmesh
    auto distance_function = distmesh::distance_function::circular(radius);
    auto mesh = distmesh::distmesh(distance_function, config["outer_edge_length"].toDouble(),
        1.0 + (1.0 - config["inner_edge_length"].toDouble() / config["outer_edge_length"].toDouble()) * distance_function / radius,
        bounding_box);

    // get boundary
    auto boundary = distmesh::boundedges(std::get<1>(mesh));

    // convert to mpflow matrix
    auto nodes_gpu = mpFlow::numeric::matrix::fromEigen<mpFlow::dtype::real, distmesh::dtype::real>(
        std::get<0>(mesh), stream);
    auto elements_gpu = mpFlow::numeric::matrix::fromEigen<mpFlow::dtype::index, distmesh::dtype::index>(
        std::get<1>(mesh), stream);
    auto boundary_gpu = mpFlow::numeric::matrix::fromEigen<mpFlow::dtype::index, distmesh::dtype::index>(
        boundary, stream);

    return std::make_tuple(nodes_gpu, elements_gpu, boundary_gpu);
}

Solver::Solver(const QJsonObject& config,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
    int parallel_images, int cuda_device, QObject *parent) :
    QObject(parent), solve_time_(0), cuda_stream_(nullptr), cublas_handle_(nullptr),
    cuda_device_(cuda_device) {
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
            // create and init solver
            this->eit_solver_ = Solver::createSolverFromConfig(config, nodes, elements,
                boundary, parallel_images, this->cublas_handle(), this->cuda_stream());
            this->eit_solver()->preSolve(this->cublas_handle(), this->cuda_stream());

        } catch (const std::exception& e) {
            success = false;
        }

        // signal solver ready
        emit this->initialized(success);
    });

    // start thread
    this->thread()->start();
}

void Solver::solve(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* data) {
    // copy data to solver
    for (mpFlow::dtype::index i = 0; i < data->size(); ++i) {
        this->eit_solver()->measurement()[i]->copy((*data)[i], this->cuda_stream());
    }
    cudaStreamSynchronize(this->cuda_stream());
    this->time().restart();

    auto solver_result = this->eit_solver()->solve_differential(
        this->cublas_handle(), this->cuda_stream());
    solver_result->copyToHost(this->cuda_stream());

    cudaStreamSynchronize(this->cuda_stream());
    this->solve_time() = this->time().elapsed();

    // convert eit solver data to Siemens
    Eigen::ArrayXXf result = this->eit_solver()->forward_solver()->model()->sigma_ref() *
        (mpFlow::numeric::matrix::toEigen<mpFlow::dtype::real>(solver_result) * std::log(10.0) / 10.0).exp();

    emit this->data_ready(result, this->repeat_time().elapsed());
    this->repeat_time().restart();
}
