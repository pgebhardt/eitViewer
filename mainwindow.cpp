#include <mpflow/mpflow.h>
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
    this->statusBar()->hide();

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
        std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::real>>(1, 1, nullptr));

    // TODO
    // init table widget
    this->initTable();

    // enable auto calibrator menu items
    if (this->hasMultiGPU()) {
        this->ui->actionAuto_Calibrate->setEnabled(true);
        this->ui->actionCalibrator_Settings->setEnabled(true);
        this->ui->actionCalibrator_Data->setEnabled(true);
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

void MainWindow::initTable() {
    this->addAnalysis("solve time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) -> mpFlow::dtype::real {
        return this->solver()->solve_time();
    });
    if (this->hasMultiGPU()) {
        this->addAnalysis("calibrate time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) -> mpFlow::dtype::real {
            return this->calibrator()->solve_time();
        });
    }
    this->addAnalysis("min:", "dB", [](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::min((*values)(i, 0), result);
        }
        return result;
    });
    this->addAnalysis("max:", "dB", [](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::max((*values)(i, 0), result);
        }
        return result;
    });
    this->addAnalysis("rms:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real rms = 0.0;
        mpFlow::dtype::real area = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            rms += mpFlow::math::square((*values)(i, 0)) * this->ui->image->element_area()[i];
            area += this->ui->image->element_area()[i];
        }
        return std::sqrt(rms / area);
    });
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

void MainWindow::addAnalysis(QString name, QString unit,
    std::function<mpFlow::dtype::real(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>)> analysis) {
    // create new table row and table items
    this->ui->analysis_table->insertRow(this->ui->analysis_table->rowCount());
    this->ui->analysis_table->setItem(this->ui->analysis_table->rowCount() - 1, 0,
        new QTableWidgetItem(name));
    this->ui->analysis_table->setItem(this->ui->analysis_table->rowCount() - 1, 1,
        new QTableWidgetItem(""));

    // add function to vector
    this->analysis().push_back(std::make_tuple(
        this->ui->analysis_table->rowCount() - 1, unit, analysis));
}

void MainWindow::draw() {
    if (this->solver()) {
        // solve
        auto gamma = this->solver()->dgamma();
        if (this->ui->actionCalibrator_Data->isChecked()) {
            gamma = this->calibrator()->gamma();
        }

        // update image
        this->ui->image->draw(gamma, this->ui->actionAuto_Normalize->isChecked());

        // evaluate analysis functions
        for (const auto& analysis : this->analysis()) {
            this->ui->analysis_table->item(std::get<0>(analysis), 1)->setText(
                QString("%1 ").arg(std::get<2>(analysis)(gamma)) + std::get<1>(analysis));
        }
    }
}

void MainWindow::on_actionOpen_triggered() {
    // get open file name
    QString file_name = QFileDialog::getOpenFileName(
        this, "Load Solver", "", "Solver File (*.conf)");

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
            connect(this->solver(), &Calibrator::initialized, this,
                &MainWindow::calibrator_initialized);
        }

        file.close();
    }
}


void MainWindow::on_actionExit_triggered() {
    // quit application
    this->close();
}

void MainWindow::on_actionLoad_Measurement_triggered() {
    if (this->solver()) {
        // get load file name
        QString file_name = QFileDialog::getOpenFileName(
            this, "Load Measurement", "", "Matrix File (*.txt)");

        if (file_name != "") {
            // load matrix
            try {
                auto voltage = mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::real>(file_name.toStdString(), nullptr);
                this->solver()->measurement()->copy(voltage, nullptr);
            } catch(const std::exception&) {
                QMessageBox::information(this, this->windowTitle(), "Cannot load measurement matrix!");
            }
        }
    }
}

void MainWindow::on_actionSave_Measurement_triggered() {
    if (this->solver()) {
        // get save file name
        QString file_name = QFileDialog::getSaveFileName(
            this, "Save Measurement", "", "Matrix File (*.txt)");

        // save measurement to file
        if (file_name != "") {
            this->solver()->measurement()->copyToHost(nullptr);
            cudaStreamSynchronize(nullptr);
            mpFlow::numeric::matrix::savetxt(file_name.toStdString(), this->solver()->measurement());
        }
    }
}

void MainWindow::on_actionCalibrate_triggered() {
    if (this->solver()) {
        // set calibration voltage to current measurment voltage
        this->solver()->calculation()->copy(this->solver()->measurement(), nullptr);
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

void MainWindow::on_actionAnalysis_Table_toggled(bool arg1) {
    this->ui->analysis_table->setVisible(arg1);
}

void MainWindow::on_actionSave_Image_triggered() {
    // get save file name
    QString file_name = QFileDialog::getSaveFileName(
        this, "Save Image", "", "PNG File (*.png)");

    // save image
    if (file_name != "") {
        // grap frame buffer
        QImage bitmap = this->ui->image->grabFrameBuffer();

        bitmap.save(file_name, "PNG");
    }
}

void MainWindow::on_actionAbout_triggered() {
    QMessageBox::about(this, this->windowTitle(), tr("Version: ") + APP_REVISION);
}

void MainWindow::solver_initialized(bool success) {
    if (success) {
        // init image
        this->draw_timer().stop();
        this->ui->image->init(this->solver()->eit_solver()->forward_solver()->model());
        this->ui->image->draw(this->solver()->eit_solver()->dgamma(),
            this->ui->actionAuto_Normalize->isChecked());
        this->draw_timer().start(20);

        // set correct matrix for measurement system with meta object method call
        // to ensure matrix update not during data read or write
        qRegisterMetaType<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>(
            "std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>");
        QMetaObject::invokeMethod(this->measurement_system(), "setMeasurementMatrix",
            Qt::AutoConnection, Q_ARG(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
                this->solver()->measurement()));
    } else {
        QMessageBox::information(this, this->windowTitle(),
            tr("Cannot load solver from config!"));
    }
}

void MainWindow::calibrator_initialized(bool success) {
    if (!success) {
        QMessageBox::information(this, this->windowTitle(),
            tr("Cannot create calibrator!"));
    }
}
