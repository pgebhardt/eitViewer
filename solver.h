#ifndef SOLVER_H
#define SOLVER_H

#include <QObject>
#include <QThread>
#include <QJsonObject>
#include <QJsonArray>
#include <mpflow/mpflow.h>
#include "highprecisiontime.h"

class Solver : public QObject {
    Q_OBJECT
public:
    explicit Solver(const QJsonObject& config,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
        int parallel_images, int cuda_device=0, QObject* parent=nullptr);

    static std::tuple<
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>>
        createMeshFromConfig(const QJsonObject& config, cudaStream_t stream);

    static std::shared_ptr<mpFlow::EIT::solver::Solver<mpFlow::numeric::Conjugate>>
        createSolverFromConfig(const QJsonObject& config,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements,
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary,
        int parallel_images, cublasHandle_t handle, cudaStream_t stream);

signals:
    void initialized(bool success);
    void data_ready(Eigen::ArrayXXf data, double time_elapsed);

public slots:
    void solve(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>* data);

public:
    // accessors
    std::shared_ptr<mpFlow::EIT::solver::Solver<
        mpFlow::numeric::Conjugate>> eit_solver() {
        return this->eit_solver_;
    }
    QThread* thread() { return this->thread_; }
    HighPrecisionTime& time() { return this->time_; }
    HighPrecisionTime& repeat_time() { return this->repeat_time_; }
    const cudaStream_t& cuda_stream() { return this->cuda_stream_; }
    const cublasHandle_t& cublas_handle() { return this->cublas_handle_; }
    int cuda_device() { return this->cuda_device_; }
    double& solve_time() { return this->solve_time_; }

private:
    // member
    std::shared_ptr<mpFlow::EIT::solver::Solver<
        mpFlow::numeric::Conjugate>> eit_solver_;
    QThread* thread_;
    HighPrecisionTime time_;
    HighPrecisionTime repeat_time_;
    double solve_time_;
    cudaStream_t cuda_stream_;
    cublasHandle_t cublas_handle_;
    int cuda_device_;
};

#endif // SOLVER_H
