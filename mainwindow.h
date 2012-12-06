#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <fasteit/fasteit.h>

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
    void on_actionLoad_Voltage_triggered();

protected:
    void createSolver();

public:
    // accessor
    const fastEIT::Solver<fastEIT::basis::Linear>& solver() const {
        return *this->solver_;
    }
    const cublasHandle_t& handle() const { return this->handle_; }

    // mutators
    fastEIT::Solver<fastEIT::basis::Linear>& solver() { return *this->solver_; }
    cublasHandle_t& handle() { return this->handle_; }

private:
    Ui::MainWindow *ui;
    fastEIT::Solver<fastEIT::basis::Linear>* solver_;
    cublasHandle_t handle_;
};

#endif // MAINWINDOW_H
