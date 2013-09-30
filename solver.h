#ifndef SOLVER_H
#define SOLVER_H

#include <QObject>
#include <QThread>
#include <QJsonObject>
#include <QJsonArray>
#include <QTime>
#include <mpflow/mpflow.h>

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

public slots:
    void solve();

public:
    // accessors
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> image() {
        return this->eit_solver()->dgamma();
    }
    std::shared_ptr<mpFlow::EIT::solver::Solver<
        mpFlow::numeric::Conjugate>> eit_solver() {
        return this->eit_solver_;
    }
    QThread* thread() { return this->thread_; }
    QTime& time() { return this->time_; }
    const cudaStream_t& cuda_stream() { return this->cuda_stream_; }
    const cublasHandle_t& cublas_handle() { return this->cublas_handle_; }
    int cuda_device() { return this->cuda_device_; }
    int& solve_time() { return this->solve_time_; }

private:
    // member
    std::shared_ptr<mpFlow::EIT::solver::Solver<
        mpFlow::numeric::Conjugate>> eit_solver_;
    QThread* thread_;
    QTime time_;
    int solve_time_;
    cudaStream_t cuda_stream_;
    cublasHandle_t cublas_handle_;
    int cuda_device_;
};

#endif // SOLVER_H
