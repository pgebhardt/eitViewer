#include <fstream>
#include <mpflow/mpflow.h>
#include <QtCore>
#include <QtGui>
#include <QInputDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"
#include "measurementsystem.h"
#include "calibratordialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::MainWindow), measurement_system_(nullptr),
    solver_(nullptr), calibrator_(nullptr), datalogger_(nullptr), open_file_name_("") {
    ui->setupUi(this);
    this->statusBar()->hide();

    // create measurement system
    this->measurement_system_ = new MeasurementSystem();

    // create data logger
    this->datalogger_ = new DataLogger();
    connect(this->ui->actionReset_DataLogger, &QAction::triggered, this->datalogger(), &DataLogger::reset_log);

    // TODO
    // init table widget
    this->initTable();
    this->analysis_timer_ = new QTimer(this);
    connect(this->analysis_timer_, &QTimer::timeout, this, &MainWindow::analyse);

    // set up environment for auto calibrator, if system has multi gpus
    if (this->hasMultiGPU()) {
        // enable peer access for 2 gpus
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        cudaSetDevice(0);

        // enable menu items
        this->ui->actionAuto_Calibrate->setEnabled(true);
        this->ui->actionCalibrator_Settings->setEnabled(true);
    }
}

MainWindow::~MainWindow() {
    // close solver
    this->close_solver();

    // cleanup measurement system
    if (this->measurement_system()) {
        this->measurement_system()->thread()->quit();
        this->measurement_system()->thread()->wait();
        delete this->measurement_system();
    }

    delete this->ui;
}

void MainWindow::initTable() {
    this->addAnalysis("system fps:", "", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return 1e3 / (20.0 / this->ui->image->image_increment());
    });
    this->addAnalysis("latency:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return 20.0 / this->ui->image->image_increment() * this->solver()->eit_solver()->measurement().size() + this->solver()->solve_time() * 1e3;
    });
    this->addAnalysis("solve time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return this->solver()->solve_time() * 1e3;
    });
    if (this->hasMultiGPU()) {
        this->addAnalysis("calibrate time:", "ms", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
            return this->calibrator()->solve_time() * 1e3;
        });
    }
    this->addAnalysis("normalization threashold:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>) {
        return this->ui->image->threashold();
    });
    this->addAnalysis("min:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::min((*values)(i, this->ui->image->image_pos()), result);
        }
        return result;
    });
    this->addAnalysis("max:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real result = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            result = std::max((*values)(i, this->ui->image->image_pos()), result);
        }
        return result;
    });
    this->addAnalysis("rms:", "dB", [=](std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) -> mpFlow::dtype::real {
        mpFlow::dtype::real rms = 0.0;
        mpFlow::dtype::real area = 0.0;
        for (mpFlow::dtype::index i = 0; i < values->rows(); ++i) {
            rms += mpFlow::math::square((*values)(i, this->ui->image->image_pos())) * this->ui->image->element_area()[i];
            area += this->ui->image->element_area()[i];
        }
        return std::sqrt(rms / area);
    });
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

void MainWindow::analyse() {
    if (this->solver()) {
        // evaluate analysis functions
        for (const auto& analysis : this->analysis()) {
            this->ui->analysis_table->item(std::get<0>(analysis), 1)->setText(
                QString("%1 ").arg(std::get<2>(analysis)(this->ui->image->data())) + std::get<1>(analysis));
        }
    }
}

void MainWindow::on_actionOpen_triggered() {
    // get open file name
    QString file_name = QFileDialog::getOpenFileName(
        this, "Load Solver", this->open_file_name(),
        "Solver File (*.conf)");

    // load solver
    if (file_name != "") {
        // close current solver
        this->close_solver();

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
        mpFlow::dtype::index parallel_images = config["solver"].toObject()["parallel_images"].toDouble();
        this->solver_ = new Solver(config, std::get<0>(mesh), std::get<1>(mesh), std::get<2>(mesh),
            parallel_images == 0 ? 1 : parallel_images, 0);
        connect(this->solver(), &Solver::initialized, this, &MainWindow::solver_initialized);

        // create auto calibrator
        if (this->hasMultiGPU()) {
            this->calibrator_ = new Calibrator(this->solver(), config,
                std::get<0>(mesh), std::get<1>(mesh), std::get<2>(mesh), 1);
            connect(this->calibrator(), &Calibrator::initialized, this,
                &MainWindow::calibrator_initialized);
        }

        // save current file name
        this->open_file_name() = file_name;
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
            this, "Load Measurement", this->open_file_name(),
             "Matrix File (*.txt)");

        if (file_name != "") {
            // load matrix
            try {
                auto voltage = mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::real>(file_name.toStdString(), nullptr);
                for (auto& measurement : this->measurement_system()->measurement_buffer()) {
                    measurement->copy(voltage, nullptr);
                }
                emit this->measurement_system()->data_ready(
                    &this->measurement_system()->measurement_buffer());
            } catch(const std::exception&) {
                QMessageBox::information(this, this->windowTitle(), "Cannot load measurement matrix!");
            }

            // save current file name
            this->open_file_name() = file_name;
        }
    }
}

void MainWindow::on_actionSave_Measurement_triggered() {
    if (this->solver()) {
        // get save file name
        QString file_name = QFileDialog::getSaveFileName(
            this, "Save Measurement", this->open_file_name(),
            "Matrix File (*.txt)");

        // save measurement to file
        if (file_name != "") {
            this->solver()->eit_solver()->measurement()[0]->copyToHost(nullptr);
            cudaStreamSynchronize(nullptr);
            mpFlow::numeric::matrix::savetxt(file_name.toStdString(),
                this->solver()->eit_solver()->measurement()[0]);
        }

        // save current file name
        this->open_file_name() = file_name;
    }
}

void MainWindow::on_actionCalibrate_triggered() {
    if (this->solver()) {
        // set calibration voltage to current measurment voltage
        for (mpFlow::dtype::index i = 0; i < this->solver()->eit_solver()->calculation().size(); ++i) {
            this->solver()->eit_solver()->calculation()[i]->copy(
                this->measurement_system()->measurement_buffer()[i], nullptr);
        }

        emit this->measurement_system()->data_ready(
            &this->measurement_system()->measurement_buffer());
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

void MainWindow::on_actionRun_DataLogger_toggled(bool arg1) {
    if (arg1) {
        this->datalogger()->start_logging();
    } else {
        this->datalogger()->stop_logging();
    }
}

void MainWindow::on_actionSave_DataLogger_triggered() {
    if (this->solver()) {
        // get save file name
        QString file_name = QFileDialog::getSaveFileName(
            this, "Save Log", "", "Log File (*.log)");

        // save mesh adn data log
        if (file_name != "") {
            // save nodes and elements
            mpFlow::numeric::matrix::savetxt((file_name + ".nodes").toStdString(),
                this->solver()->eit_solver()->forward_solver()->model()->mesh()->nodes());
            mpFlow::numeric::matrix::savetxt((file_name + ".elements").toStdString(),
                this->solver()->eit_solver()->forward_solver()->model()->mesh()->elements());

            // save log
            std::ofstream file;
            file.open(file_name.toStdString().c_str());
            if (file.fail()) {
                QMessageBox::information(this, this->windowTitle(),
                    tr("Cannot save log!"));
            }

            this->datalogger()->dump(&file);
            file.close();
        }
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
        this->ui->image->init(this->solver()->eit_solver()->forward_solver()->model(),
            this->solver()->eit_solver()->dgamma()->rows(),
            this->solver()->eit_solver()->dgamma()->columns());
        qRegisterMetaType<std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>>(
            "std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>");
        connect(this->solver(), &Solver::data_ready, this->ui->image, &Image::update_data);

        // set correct matrix for measurement system with meta object method call
        // to ensure matrix update not during data read or write
        qRegisterMetaType<mpFlow::dtype::index>("mpFlow::dtype::index");
        QMetaObject::invokeMethod(this->measurement_system(), "init", Qt::AutoConnection,
            Q_ARG(mpFlow::dtype::index, this->solver()->eit_solver()->measurement().size()),
            Q_ARG(mpFlow::dtype::index, this->solver()->eit_solver()->measurement()[0]->rows()),
            Q_ARG(mpFlow::dtype::index, this->solver()->eit_solver()->measurement()[0]->columns()));
        connect(this->measurement_system(), &MeasurementSystem::data_ready, this->solver(), &Solver::solve);
        connect(this->solver(), &Solver::data_ready, this->ui->image, &Image::update_data);
        connect(this->solver(), &Solver::data_ready, this->datalogger(), &DataLogger::add_data);

        // TODO
        this->analysis_timer_->start(20);
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

void MainWindow::close_solver() {
    // stop drawing image
    this->ui->image->cleanup();
    this->analysis_timer_->stop();

    // stop and cleanup auto calibartor
    if (this->calibrator()) {
        this->calibrator()->thread()->quit();
        this->calibrator()->thread()->wait();
        delete this->calibrator();
        this->calibrator_ = nullptr;
    }

    // stop and cleanup solver
    if (this->solver()) {
        disconnect(this->measurement_system(), &MeasurementSystem::data_ready, this->solver(), &Solver::solve);
        this->solver()->thread()->quit();
        this->solver()->thread()->wait();
        delete this->solver();
        this->solver_ = nullptr;
    }
}
