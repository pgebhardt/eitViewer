#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QTime>
#include <QLabel>
#include <fasteit/fasteit.h>
#include "measurementsystem.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:
    void draw();
    void measurementSystemConnectionError(QAbstractSocket::SocketError socket_error);

    void on_actionLoad_Voltage_triggered();

    void on_actionStart_Solver_triggered();

    void on_actionStop_Solver_triggered();

    void on_actionCalibrate_triggered();

    void on_actionSave_Image_triggered();

    void on_actionSave_Voltage_triggered();

    void on_actionConnect_triggered();

    void on_actionDisconnect_triggered();

protected:
    void createSolver();
    void createStatusBar();

public:
    // accessor
    const std::shared_ptr<fastEIT::Solver<fastEIT::basis::Linear>> solver() const {
        return this->solver_;
    }
    const MeasurementSystem& measurement_system() const { return *this->measurement_system_; }
    const cublasHandle_t& handle() const { return this->handle_; }
    const QTimer& draw_timer() const { return *this->draw_timer_; }
    const QTime& time() const { return this->time_; }
    const QLabel& fps_label() const { return *this->fps_label_; }
    const QLabel& min_label() const { return *this->min_label_; }
    const QLabel& max_label() const { return *this->max_label_; }

    // mutators
    std::shared_ptr<fastEIT::Solver<fastEIT::basis::Linear>> solver() { return this->solver_; }
    MeasurementSystem& measurement_system() { return *this->measurement_system_; }
    cublasHandle_t& handle() { return this->handle_; }
    QTimer& draw_timer() { return *this->draw_timer_; }
    QTime& time() { return this->time_; }
    QLabel& fps_label() { return *this->fps_label_; }
    QLabel& min_label() { return *this->min_label_; }
    QLabel& max_label() { return *this->max_label_; }

private:
    Ui::MainWindow *ui;
    MeasurementSystem* measurement_system_;
    std::shared_ptr<fastEIT::Solver<fastEIT::basis::Linear>> solver_;
    cublasHandle_t handle_;
    QTimer* draw_timer_;
    QTime time_;
    QLabel* fps_label_;
    QLabel* min_label_;
    QLabel* max_label_;
};

#endif // MAINWINDOW_H
