#include <fasteit/fasteit.h>

#include <QtCore>
#include <QtGui>
#include <QHostAddress>
#include <QInputDialog>
#include <QMessageBox>
#include <iostream>
#include <sstream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"
#include "measurementsystem.h"
#include "jsonobject.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) {
    ui->setupUi(this);

    // create timer
    this->draw_timer_ = new QTimer(this);
    connect(this->draw_timer_, SIGNAL(timeout()), this, SLOT(draw()));

    // create cublas handle
    cublasCreate(&this->handle_);

    // create measurement system
    auto measurement = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(1, 1, nullptr);
    this->measurement_system_ = new MeasurementSystem(nullptr, measurement);

    // create status bar
    this->createStatusBar();
}

MainWindow::~MainWindow() {
    delete this->ui;
    cublasDestroy(this->handle_);
}

void MainWindow::createStatusBar() {
    // create label
    this->fps_label_ = new QLabel("fps:", this);
    this->min_label_ = new QLabel("min:", this);
    this->max_label_ = new QLabel("max:", this);

    // set frame style
    this->fps_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->min_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);
    this->max_label().setFrameStyle(QFrame::Panel | QFrame::Sunken);

    // fill status bar
    this->statusBar()->addPermanentWidget(&this->fps_label(), 1);
    this->statusBar()->addPermanentWidget(&this->min_label(), 1);
    this->statusBar()->addPermanentWidget(&this->max_label(), 1);
}

void MainWindow::draw() {
    // check for image
    if (this->image()) {
        // copy data to device
        this->solver()->measured_voltage()->copyToDevice(nullptr);

        // solve
        auto gamma = this->solver()->solve(this->handle(), NULL);
        gamma->copyToHost(NULL);

        // cut values
        for (fastEIT::dtype::index element = 0; element < gamma->rows(); ++element) {
            if ((!this->ui->actionShow_Negative_Values->isChecked()) && ((*gamma)(element, 0) < 0.0)) {
                (*gamma)(element, 0) = 0.0;
            }
            if ((!this->ui->actionShow_Positive_Values->isChecked()) && ((*gamma)(element, 0) > 0.0)) {
                (*gamma)(element, 0) = 0.0;
            }
        }

        // update image
        fastEIT::dtype::real min_value, max_value;
        std::tie(min_value, max_value) = this->image()->draw(gamma,
            this->ui->actionShow_Transparent_Values->isChecked(),
            true);

        // calc fps
        this->fps_label().setText(QString("fps: %1").arg(1e3 / this->time().elapsed()));
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

        // copy voltage
        this->solver()->measured_voltage()->copy(voltage, nullptr);
        this->solver()->measured_voltage()->copyToHost(nullptr);
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

void MainWindow::on_actionStart_Solver_triggered() {
    // start timer
    this->draw_timer().start(30);
}

void MainWindow::on_actionStop_Solver_triggered() {
    // stop timer
    this->draw_timer().stop();
}

void MainWindow::on_actionCalibrate_triggered() {
    // set calibration voltage to current measurment voltage
    this->solver()->calibration_voltage()->copy(this->solver()->measured_voltage(), NULL);
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

        try {
            // parse to json object
            JsonObject config(QString(file.readAll()).trimmed());

            // get configs
            JsonObject solver_config = config.getObject("solver");
            JsonObject model_config = config.getObject("model");

            // load mesh strings
            std::stringstream nodes_stream(model_config.getObject("mesh").getString("nodes").toStdString());
            std::stringstream elements_stream(model_config.getObject("mesh").getString("elements").toStdString());
            std::stringstream boundary_stream(model_config.getObject("mesh").getString("boundary").toStdString());

            // load pattern strings
            std::stringstream drive_pattern_stream(model_config.getObject("source").getString("drive_pattern").toStdString());
            std::stringstream measurement_pattern_stream(model_config.getObject("source").getString("measurement_pattern").toStdString());

            // load mesh matrices
            auto nodes = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(&nodes_stream, nullptr);
            auto elements = fastEIT::matrix::loadtxt<fastEIT::dtype::index>(&elements_stream, nullptr);
            auto boundary = fastEIT::matrix::loadtxt<fastEIT::dtype::index>(&boundary_stream, nullptr);

            // load pattern matrices
            auto drive_pattern = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(&drive_pattern_stream, nullptr);
            auto measurement_pattern = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(&measurement_pattern_stream, nullptr);

            // create mesh
            auto mesh = fastEIT::mesh::quadraticBasis(nodes, elements, boundary,
                model_config.getObject("mesh").getDouble("radius"),
                model_config.getObject("mesh").getDouble("height"), nullptr);

            // create electrodes
            auto electrodes = fastEIT::electrodes::circularBoundary(
                model_config.getObject("electrodes").getInt("count"),
                std::make_tuple(model_config.getObject("electrodes").getDouble("width"),
                    model_config.getObject("electrodes").getDouble("height")),
                1.0, mesh->radius());

            // create source
            auto source = std::make_shared<fastEIT::source::Current<fastEIT::basis::Quadratic>>(
                model_config.getObject("source").getDouble("current"), mesh, electrodes,
                model_config.getInt("components_count"), drive_pattern, measurement_pattern,
                this->handle(), nullptr);

            // create model
            auto model = std::make_shared<fastEIT::Model<fastEIT::basis::Quadratic>>(
                mesh, electrodes, source, model_config.getDouble("sigma_ref"),
                model_config.getInt("components_count"), this->handle(), nullptr);

            // create and init solver
            this->solver_ = std::make_shared<fastEIT::Solver>(model,
                solver_config.getDouble("regularization_factor"), this->handle(), nullptr);
            this->solver()->preSolve(this->handle(), nullptr);
            this->solver()->measured_voltage()->copyToHost(nullptr);

            // create image
            this->image_ = new Image(this->solver()->model());
            this->image()->draw(this->solver()->dgamma(),
                this->ui->actionShow_Transparent_Values->isChecked(),
                true);
            this->setCentralWidget(this->image());

            // set correct matrix for measurement system
            this->measurement_system().setMeasurementMatrix(this->solver()->measured_voltage());

        } catch (std::exception& e) {
            QMessageBox::information(this, this->windowTitle(),
                                     tr("Cannot load solver config!"));
        }
    }
}
void MainWindow::on_actionExit_triggered() {
    // quit application
    this->close();
}

