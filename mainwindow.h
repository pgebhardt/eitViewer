#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QTime>
#include <QLabel>
#include <fasteit/fasteit.h>
#include "image.h"
#include "measurementsystem.h"
#include "solver.h"

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

    void on_actionLoad_Voltage_triggered();

    void on_actionCalibrate_triggered();

    void on_actionSave_Image_triggered();

    void on_actionSave_Voltage_triggered();

    void on_actionExit_triggered();

    void on_actionOpen_triggered();

    void on_solver_initialized();

protected:
    void createStatusBar();

public:
    // accessor
    MeasurementSystem* measurement_system() { return this->measurement_system_; }
    Solver* solver() { return this->solver_; }
    Image* image() { return this->image_; }
    QTimer& draw_timer() { return *this->draw_timer_; }
    QTime& time() { return this->time_; }
    QLabel& fps_label() { return *this->fps_label_; }
    QLabel& solve_time_label() { return *this->solve_time_label_; }
    QLabel& min_label() { return *this->min_label_; }
    QLabel& max_label() { return *this->max_label_; }

private:
    Ui::MainWindow *ui;
    MeasurementSystem* measurement_system_;
    Solver* solver_;
    Image* image_;
    cublasHandle_t handle_;
    QTimer* draw_timer_;
    QTime time_;
    QLabel* fps_label_;
    QLabel* solve_time_label_;
    QLabel* min_label_;
    QLabel* max_label_;
};

#endif // MAINWINDOW_H
