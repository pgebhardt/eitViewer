#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
