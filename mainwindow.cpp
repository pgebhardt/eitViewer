#include <fstream>
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
    solver_(nullptr), calibrator_(nullptr), image_pos_(0.0), image_increment_(0.0) {
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
    this->measurement_system_ = new MeasurementSystem();

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
    // cleanup solver
    this->cleanupSolver();

    // cleanup measurement system
    if (this->measurement_system()) {
        this->measurement_system()->thread()->quit();
        this->measurement_system()->thread()->wait();
        delete this->measurement_system();
    }

    delete this->ui;
}

void MainWindow::initTable() {
    this->addAnalysis("solve time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return this->solver()->solve_time();
    });
    if (this->hasMultiGPU()) {
        this->addAnalysis("calibrate time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
            return this->calibrator()->solve_time();
        });
    }
    this->addAnalysis("normalization threashold:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return this->ui->image->threashold();
    });
    this->addAnalysis("min:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::min((*values)(i, this->image_pos()), result);
        }
        return result;
    });
    this->addAnalysis("max:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::max((*values)(i, this->image_pos()), result);
        }
        return result;
    });
    this->addAnalysis("rms:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real rms = 0.0;
        mpFlow::dtype::real area = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            rms += mpFlow::math::square((*values)(i, this->image_pos())) * this->ui->image->element_area()[i];
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
        disconnect(this->measurement_system(), &MeasurementSystem::data_ready, this->solver(), &Solver::solve);
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
        auto image = this->solver()->image();
/*        if (this->ui->actionCalibrator_Data->isChecked()) {
            image = this->calibrator()->image();

            // update image
            this->ui->image->draw(image);
        } else {*/
            // update image
            this->ui->image->draw(image, (int)this->image_pos());
            this->image_pos() += this->image_increment();
            if (this->image_pos() >= (double)image->columns()) {
                this->image_pos() = (double)image->columns() - 1.0;
                this->image_increment() = 0.0;
            }
//        }

        // evaluate analysis functions
        for (const auto& analysis : this->analysis()) {
            this->ui->analysis_table->item(std::get<0>(analysis), 1)->setText(
                QString("%1 ").arg(std::get<2>(analysis)(image)) + std::get<1>(analysis));
        }

        if (this->calibrator()->running()) {
            std::ofstream file;
            file.open("out/rms.txt", std::ofstream::out | std::ofstream::app);
            file << std::get<2>(this->analysis()[5])(image) << std::endl;
            file.close();
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
        file.close();

        // create same mesh for both solver and calibrator
        auto mesh = Solver::createMeshFromConfig(
            config["model"].toObject()["mesh"].toObject(), nullptr);

        // create new Solver from config
        this->solver_ = new Solver(config, std::get<0>(mesh), std::get<1>(mesh),
            std::get<2>(mesh), 16, 0);
        connect(this->solver(), &Solver::initialized, this, &MainWindow::solver_initialized);

        // create auto calibrator
        if (this->hasMultiGPU()) {
            this->calibrator_ = new Calibrator(this->solver(), config,
                std::get<0>(mesh), std::get<1>(mesh), std::get<2>(mesh), 1);
            connect(this->calibrator(), &Calibrator::initialized, this,
                &MainWindow::calibrator_initialized);
        }
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
                this->measurement_system()->measurement_buffer()[0]->copy(voltage, nullptr);
                this->measurement_system()->buffer_pos() = 0;
                emit this->measurement_system()->data_ready();
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
            this->measurement_system()->measurement_buffer()[0]->copyToHost(nullptr);
            cudaStreamSynchronize(nullptr);
            mpFlow::numeric::matrix::savetxt(file_name.toStdString(),
                this->measurement_system()->measurement_buffer()[0]);
        }
    }
}

void MainWindow::on_actionCalibrate_triggered() {
    if (this->solver()) {
        // set calibration voltage to current measurment voltage
        for (mpFlow::dtype::index i = 0; i < this->solver()->eit_solver()->calculation().size(); ++i) {
            this->solver()->eit_solver()->calculation()[i]->copy(
                this->measurement_system()->measurement_buffer()[i], nullptr);
        }

        this->measurement_system()->buffer_pos() = 0;
        emit this->measurement_system()->data_ready();
    }
}

void MainWindow::on_actionAuto_Calibrate_toggled(bool arg1) {
    if (this->calibrator()) {
        if (arg1) {
            QMetaObject::invokeMethod(this->calibrator(), "start",
                Qt::AutoConnection, Q_ARG(int, this->calibrator()->step_size()));
        } else {
            QMetaObject::invokeMethod(this->calibrator(), "stop");
        }
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

void MainWindow::on_actionVersion_triggered() {
    // Show about box with version number
    QMessageBox::about(this, this->windowTitle(), tr("%1: %2\nmpFlow: %3").arg(
        this->windowTitle(), GIT_VERSION, mpFlow::version::getVersionString()));
}

void MainWindow::solver_initialized(bool success) {
    if (success) {
        // init image
        this->draw_timer().stop();
        this->ui->image->init(this->solver()->eit_solver()->forward_solver()->model());
        this->ui->image->draw(this->solver()->eit_solver()->dgamma());
        this->draw_timer().start(20);

        // set correct matrix for measurement system with meta object method call
        // to ensure matrix update not during data read or write
        qRegisterMetaType<std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>*>(
            "std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>*");
        QMetaObject::invokeMethod(this->measurement_system(), "setMeasurementBuffer",
            Qt::AutoConnection, Q_ARG(std::vector<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>*,
                &this->solver()->eit_solver()->measurement()));
        connect(this->measurement_system(), &MeasurementSystem::data_ready, this->solver(), &Solver::solve);
        connect(this->solver(), &Solver::data_ready, this, &MainWindow::update_image_increment);
    } else {
        QMessageBox::information(this, this->windowTitle(),
            tr("Cannot load solver from config!"));
    }
}

void MainWindow::calibrator_initialized(bool success) {
    if (!success) {
        QMessageBox::information(this, this->windowTitle(),
            tr("Cannot create calibrator!"));
    } else {
        // connect(this->calibrator()->timer(), &QTimer::timeout, this->solver(), &Solver::solve);
    }
}

void MainWindow::update_image_increment(int time_elapsed) {
    this->image_pos() = 0.0;
    this->image_increment() = time_elapsed > 20 ? 20.0 / (double)time_elapsed *
        (double)this->solver()->eit_solver()->measurement().size() : 0.0;
}
