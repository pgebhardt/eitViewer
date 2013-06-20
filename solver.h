#ifndef SOLVER_H
#define SOLVER_H

#include <QObject>
#include <QThread>
#include <QJsonObject>
#include <QJsonArray>
#include <QTimer>
#include <QTime>
#include <mpflow/mpflow.h>

class Solver : public QObject {
    Q_OBJECT
public:
    explicit Solver(const QJsonObject& config, int cuda_device=0,
        QObject* parent=nullptr);
    void restart(int step_size);

signals:
    void initialized(bool success);

protected slots:
    virtual void solve();

public:
    // accessors
    std::shared_ptr<mpFlow::Matrix<mpFlow::dtype::real>> measurement() {
        return this->eit_solver()->measurement(0);
    }
    std::shared_ptr<mpFlow::Matrix<mpFlow::dtype::real>> calculation() {
        return this->eit_solver()->calculation(0);
    }
    std::shared_ptr<mpFlow::Matrix<mpFlow::dtype::real>> gamma() {
        return this->eit_solver()->gamma();
    }
    std::shared_ptr<mpFlow::Matrix<mpFlow::dtype::real>> dgamma() {
        return this->eit_solver()->dgamma();
    }
    std::shared_ptr<mpFlow::EIT::solver::Solver<mpFlow::numeric::SparseConjugate,
        mpFlow::numeric::FastConjugate>> eit_solver() {
        return this->eit_solver_;
    }
    QThread* thread() { return this->thread_; }
    QTimer* solve_timer() { return this->solve_timer_; }
    QTime& time() { return this->time_; }
    int& solve_time() { return this->solve_time_; }
    const cudaStream_t& cuda_stream() { return this->cuda_stream_; }
    const cublasHandle_t& cublas_handle() { return this->cublas_handle_; }
    int cuda_device() { return this->cuda_device_; }
    int step_size() { return this->step_size_; }

private:
    // member
    std::shared_ptr<mpFlow::EIT::solver::Solver<mpFlow::numeric::SparseConjugate,
        mpFlow::numeric::FastConjugate>> eit_solver_;
    QThread* thread_;
    QTimer* solve_timer_;
    QTime time_;
    int solve_time_;
    cudaStream_t cuda_stream_;
    cublasHandle_t cublas_handle_;
    int cuda_device_;
    int step_size_;
};

#endif // SOLVER_H
