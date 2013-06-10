#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QTime>
#include <QLabel>
#include <functional>
#include <fasteit/fasteit.h>
#include "image.h"
#include "measurementsystem.h"
#include "solver.h"
#include "calibrator.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:
    void draw();
    void on_actionOpen_triggered();
    void on_actionExit_triggered();
    void on_actionLoad_Measurement_triggered();
    void on_actionSave_Measurement_triggered();
    void on_actionCalibrate_triggered();
    void on_actionAuto_Calibrate_toggled(bool arg1);
    void on_actionCalibrator_Settings_triggered();
    void on_actionSave_Image_triggered();
    void solver_initialized(bool success);
    void calibrator_initialized(bool success);

protected:
    void initTable();
    bool hasMultiGPU() { int devCount = 0; cudaGetDeviceCount(&devCount); return devCount > 1; }
    void cleanupSolver();
    void addAnalysis(QString name, QString unit, std::function<fastEIT::dtype::real(
        std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>)> analysis);

public:
    // accessor
    MeasurementSystem* measurement_system() { return this->measurement_system_; }
    Solver* solver() { return this->solver_; }
    Calibrator* calibrator() { return this->calibrator_; }
    QTimer& draw_timer() { return *this->draw_timer_; }
    std::vector<std::tuple<int, QString,
        std::function<fastEIT::dtype::real(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>)>>>&
        analysis() { return this->analysis_; }

private:
    Ui::MainWindow *ui;
    MeasurementSystem* measurement_system_;
    Solver* solver_;
    Calibrator* calibrator_;
    QTimer* draw_timer_;
    std::vector<std::tuple<int, QString,
        std::function<fastEIT::dtype::real(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>)>>>
        analysis_;
};

#endif // MAINWINDOW_H
