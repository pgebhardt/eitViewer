#include <fasteit/fasteit.h>

#include <QtCore>
#include <QtGui>
#include <QHostAddress>
#include <QInputDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <iostream>
#include <sstream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"
#include "measurementsystem.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);

    // create timer
    this->draw_timer_ = new QTimer(this);
    connect(this->draw_timer_, &QTimer::timeout, this, &MainWindow::draw);

    // create measurement system
    this->measurement_system_ = new MeasurementSystem(
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(1, 1, nullptr));

    // create status bar
    this->createStatusBar();
}

MainWindow::~MainWindow() {
    delete this->ui;

    // stop threads
    if (this->measurement_system()) {
        this->measurement_system()->thread()->quit();
        this->measurement_system()->thread()->wait();
    }
    if (this->solver()) {
        this->solver()->thread()->quit();
        this->solver()->thread()->wait();
    }
}

void MainWindow::createStatusBar() {
    // create label
    this->fps_label_ = new QLabel("fps:", this);
    this->solve_time_label_ = new QLabel("solve time:", this);
    this->min_label_ = new QLabel("min:", this);
    this->max_label_ = new QLabel("max:", this);

    // set frame style
    this->fps_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->solve_time_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->min_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->max_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);

    // fill status bar
    this->statusBar()->addPermanentWidget(&this->fps_label(), 1);
    this->statusBar()->addPermanentWidget(&this->solve_time_label(), 1);
    this->statusBar()->addPermanentWidget(&this->min_label(), 1);
    this->statusBar()->addPermanentWidget(&this->max_label(), 1);
}

void MainWindow::draw() {
    // check for image
    if (this->image()) {
        // solve
        auto gamma = this->solver()->fasteit_solver()->dgamma();

        // cut values
        if (!this->ui->actionShow_Negative_Values->isChecked()) {
            for (fastEIT::dtype::index element = 0; element < gamma->rows(); ++element) {
                if ((*gamma)(element, 0) < 0.0) {
                    (*gamma)(element, 0) = 0.0;
                }
            }
        }
        if (!this->ui->actionShow_Positive_Values->isChecked()) {
            for (fastEIT::dtype::index element = 0; element < gamma->rows(); ++element) {
                if ((*gamma)(element, 0) > 0.0) {
                    (*gamma)(element, 0) = 0.0;
                }
            }
        }

        // update image
        fastEIT::dtype::real min_value, max_value;
        std::tie(min_value, max_value) = this->image()->draw(gamma,
            this->ui->actionShow_Transparent_Values->isChecked(),
            true);

        // calc fps
        this->fps_label().setText(QString("fps: %1").arg(1e3 / this->time().elapsed()));
        this->solve_time_label().setText(QString("solve time: %1 ms").arg(this->solver()->solve_time()));
        this->solve_time_label().setText(QString("solve time: %1 ms").arg(this->solver()->solve_time()));
        this->time().restart();

        // update min max label
        this->min_label().setText(QString("min: %1 dB").arg(min_value));
        this->max_label().setText(QString("max: %1 dB").arg(max_value));
    }
}

void MainWindow::on_actionLoad_Voltage_triggered() {
    // get load file name
    QString file_name = QFileDialog::getOpenFileName(this, "Load Voltage",
                                                        "", "Matrix File (*.txt)");

    if (file_name != "") {
        // load matrix
        auto voltage = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(file_name.toStdString(), nullptr);
        this->solver()->measured_voltage()->copy(voltage, nullptr);
    }
}

void MainWindow::on_actionSave_Voltage_triggered() {
    // get save file name
    QString file_name = QFileDialog::getSaveFileName(this, "Save Voltage", "",
                                                     "Matrix File (*.txt)");

    // save voltage
    if (file_name != "") {
        this->solver()->measured_voltage()->copyToHost(nullptr);
        cudaStreamSynchronize(nullptr);
        fastEIT::matrix::savetxt(file_name.toStdString(), this->solver()->measured_voltage());
    }
}

void MainWindow::on_actionCalibrate_triggered() {
    // set calibration voltage to current measurment voltage
    this->solver()->calculated_voltage()->copy(this->solver()->measured_voltage(), nullptr);
}

void MainWindow::on_actionSave_Image_triggered() {
    // check for image
    if (this->image()) {
        // grap frame buffer
        QImage bitmap = this->image()->grabFrameBuffer();

        // get save file name
        QString file_name = QFileDialog::getSaveFileName(this, "Save Image", "",
                                                        "PNG File (*.png)");

        // save image
        if (file_name != "") {
            bitmap.save(file_name, "PNG");
        }
    }
}

void MainWindow::on_actionOpen_triggered() {
    // get open file name
    QString file_name = QFileDialog::getOpenFileName(this, "Load Solver", "",
                                                     "Solver File (*.conf)");

    // load solver
    if (file_name != "") {
        // open file
        QFile file(file_name);
        file.open(QIODevice::ReadOnly | QIODevice::Text);

        // read json config
        QString str;
        str = file.readAll();
        auto json_document = QJsonDocument::fromJson(str.toUtf8());
        auto config = json_document.object();

        // create new Solver from config
        this->solver_ = new Solver(config, 0);
        connect(this->solver_, &Solver::initialized, this, &MainWindow::solver_initialized);

        file.close();
    }
}

void MainWindow::on_actionExit_triggered() {
    // quit application
    this->close();
}

void MainWindow::solver_initialized(bool success) {
    if (success) {
        // create image
        this->image_ = new Image(this->solver()->fasteit_solver()->model());
        this->image()->draw(this->solver()->fasteit_solver()->dgamma(),
            this->ui->actionShow_Transparent_Values->isChecked(),
            true);
        this->setCentralWidget(this->image());
        this->draw_timer().start(20);

        // set correct matrix for measurement system
        this->measurement_system()->setMeasurementMatrix(this->solver()->measured_voltage());
    } else {
        QMessageBox::information(this, this->windowTitle(),
            tr("Cannot load solver from config!"));
    }
}

