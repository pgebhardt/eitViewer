#ifndef SOLVER_H
#define SOLVER_H

#include <QObject>
#include <QThread>
#include <QJsonObject>
#include <QJsonArray>
#include <QTimer>
#include <QTime>
#include <fasteit/fasteit.h>

class Solver : public QObject {
    Q_OBJECT
public:
    explicit Solver(const QJsonObject& config, QObject* parent=nullptr);

signals:
    void initialized();

public:
    // accessors
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> measured_voltage() {
        return this->fasteit_solver()->measured_voltage();
    }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> calibration_voltage() {
        return this->fasteit_solver()->calibration_voltage();
    }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> gamma() {
        return this->fasteit_solver()->gamma();
    }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> dgamma() {
        return this->fasteit_solver()->dgamma();
    }
    std::shared_ptr<fastEIT::Solver> fasteit_solver() {
        return this->fasteit_solver_;
    }
    QThread* thread() { return this->thread_; }
    cublasHandle_t handle() { return this->handle_; }
    QTimer* timer() { return this->timer_; }
    QTime& time() { return this->time_; }
    int& solve_time() { return this->solve_time_; }

private:
    // member
    std::shared_ptr<fastEIT::Solver> fasteit_solver_;
    QThread* thread_;
    cublasHandle_t handle_;
    QTimer* timer_;
    QTime time_;
    int solve_time_;
};

#endif // SOLVER_H
