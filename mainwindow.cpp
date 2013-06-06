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
#include "calibratordialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::MainWindow), measurement_system_(nullptr),
    solver_(nullptr), calibrator_(nullptr) {
    ui->setupUi(this);

    // enable peer access for 2 gpus
    if (this->hasMultiGPU()) {
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        cudaSetDevice(0);
    }

    // create timer
    this->draw_timer_ = new QTimer(this);
    connect(this->draw_timer_, &QTimer::timeout, this, &MainWindow::draw);

    // create measurement system
    this->measurement_system_ = new MeasurementSystem(
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(1, 1, nullptr));

    // create status bar
    this->createStatusBar();

    // enable auto calibrator menu items
    if (this->hasMultiGPU()) {
        this->ui->actionAuto_Calibrate->setEnabled(true);
        this->ui->actionCalibrator_Settings->setEnabled(true);
    }
}

MainWindow::~MainWindow() {
    delete this->ui;

    // cleanup measurement system
    if (this->measurement_system()) {
        this->measurement_system()->thread()->quit();
        this->measurement_system()->thread()->wait();
        delete this->measurement_system();
    }

    // cleanup solver
    this->cleanupSolver();
}

void MainWindow::createStatusBar() {
    // create label
    this->solve_time_label_ = new QLabel("solve time:", this);
    this->calibrate_time_label_ = new QLabel("calibrate time:", this);
    this->min_label_ = new QLabel("min:", this);
    this->max_label_ = new QLabel("max:", this);

    // set frame style
    this->solve_time_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->calibrate_time_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->min_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->max_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);

    // fill status bar
    this->statusBar()->addPermanentWidget(&this->solve_time_label(), 1);
    this->statusBar()->addPermanentWidget(&this->calibrate_time_label(), 1);
    this->statusBar()->addPermanentWidget(&this->min_label(), 1);
    this->statusBar()->addPermanentWidget(&this->max_label(), 1);
}

void MainWindow::cleanupSolver() {
    if (this->calibrator()) {
        this->calibrator()->thread()->quit();
        this->calibrator()->thread()->wait();
        delete this->calibrator();
        this->calibrator_ = nullptr;
    }
    if (this->solver()) {
        this->solver()->thread()->quit();
        this->solver()->thread()->wait();
        delete this->solver();
        this->solver_ = nullptr;
    }
}

void MainWindow::draw() {
    // check for image
    if (this->image()) {
        // solve
        auto gamma = this->solver()->dgamma();
        if (this->ui->actionCalibrator_Image->isChecked()) {
            gamma = this->calibrator()->gamma();
        }

        // update image
        fastEIT::dtype::real min_value, max_value;
        std::tie(min_value, max_value) = this->image()->draw(gamma,
            this->ui->actionTransparent_Values->isChecked(),
            true);

        // calc fps
        this->solve_time_label().setText(
            QString("solve time: %1 ms").arg(this->solver()->solve_time()));
        if (this->calibrator()) {
            this->calibrate_time_label().setText(
                QString("calibrate time: %1 ms").arg(this->calibrator()->solve_time()));
        }

        // update min max label
        this->min_label().setText(QString("min: %1 dB").arg(min_value));
        this->max_label().setText(QString("max: %1 dB").arg(max_value));
    }
}

void MainWindow::on_actionOpen_triggered() {
    // get open file name
    QString file_name = QFileDialog::getOpenFileName(this, "Load Solver", "",
                                                     "Solver File (*.conf)");

    // load solver
    if (file_name != "") {
        // stop drawing image
        this->draw_timer().stop();

        // cleanup old solver
        this->cleanupSolver();

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
        connect(this->solver(), &Solver::initialized, this, &MainWindow::solver_initialized);

        // create auto calibrator
        if (this->hasMultiGPU()) {
            this->calibrator_ = new Calibrator(this->solver(), config, 1);
        } else {
            this->calibrator_ = nullptr;
        }

        file.close();
    }
}

void MainWindow::on_actionExit_triggered() {
    // quit application
    this->close();
}

void MainWindow::on_actionLoad_Voltage_triggered() {
    if (this->solver()) {
        // get load file name
        QString file_name = QFileDialog::getOpenFileName(
            this, "Load Voltage", "", "Matrix File (*.txt)");

        if (file_name != "") {
            // load matrix
            auto voltage = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(file_name.toStdString(), nullptr);
            this->solver()->measured_voltage()->copy(voltage, nullptr);
        }
    }
}

void MainWindow::on_actionSave_Voltage_triggered() {
    if (this->solver()) {
        // get save file name
        QString file_name = QFileDialog::getSaveFileName(
            this, "Save Voltage", "", "Matrix File (*.txt)");

        // save voltage
        if (file_name != "") {
            this->solver()->measured_voltage()->copyToHost(nullptr);
            cudaStreamSynchronize(nullptr);
            fastEIT::matrix::savetxt(file_name.toStdString(), this->solver()->measured_voltage());
        }
    }
}

void MainWindow::on_actionCalibrate_triggered() {
    if (this->solver()) {
        // set calibration voltage to current measurment voltage
        this->solver()->calculated_voltage()->copy(this->solver()->measured_voltage(), nullptr);
    }
}

void MainWindow::on_actionAuto_Calibrate_toggled(bool arg1) {
    if (this->calibrator()) {
        this->calibrator()->running() = arg1;
    }
}

void MainWindow::on_actionCalibrator_Settings_triggered() {
    if (this->calibrator()) {
        CalibratorDialog dialog(this->calibrator(), this);
        dialog.exec();
    }
}

void MainWindow::on_actionSave_Image_triggered() {
    // check for image
    if (this->image()) {
        // get save file name
        QString file_name = QFileDialog::getSaveFileName(
            this, "Save Image", "", "PNG File (*.png)");

        // save image
        if (file_name != "") {
            // grap frame buffer
            QImage bitmap = this->image()->grabFrameBuffer();

            bitmap.save(file_name, "PNG");
        }
    }
}

void MainWindow::solver_initialized(bool success) {
    if (success) {
        // create image
        this->image_ = new Image(this->solver()->fasteit_solver()->model(), this);
        this->image()->draw(this->solver()->fasteit_solver()->dgamma(),
            this->ui->actionTransparent_Values->isChecked(),
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
