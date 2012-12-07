#include <fasteit/fasteit.h>

#include <QtCore>
#include <QtGui>
#include <QHostAddress>
#include <QInputDialog>
#include <QMessageBox>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"
#include "measurementsystem.h"

// create pattern
fastEIT::Matrix<fastEIT::dtype::real>* createPattern(fastEIT::dtype::size electrodes_count,
    fastEIT::dtype::index first_element, fastEIT::dtype::index step_width,
        cudaStream_t stream) {
    // create matrix
    auto pattern = new fastEIT::Matrix<fastEIT::dtype::real>(electrodes_count,
        electrodes_count / step_width, stream);

    // fill pattern
    for (fastEIT::dtype::index i = 0; i < pattern->columns(); ++i) {
        (*pattern)((first_element + (i + 0) * step_width) % electrodes_count, i) =
            (fastEIT::dtype::real)((i + 1) % 2) * 2.0 - 1.0;
        (*pattern)((first_element + (i + 1) * step_width) % electrodes_count, i) =
            (fastEIT::dtype::real)(i % 2) * 2.0 - 1.0;
    }

    // upload pattern
    pattern->copyToDevice(stream);

    return pattern;
}

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
    this->measurement_system_ = new MeasurementSystem(this);
    connect(&this->measurement_system(), SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(measurementSystemConnectionError(QAbstractSocket::SocketError)));

    // create solver
    this->createSolver();

    // create image
    this->setCentralWidget(new Image(this->solver().forward_solver().model().mesh(),
                                     this->solver().forward_solver().model().electrodes()));

    // create status bar
    this->createStatusBar();
}

MainWindow::~MainWindow() {
    delete this->ui;
    delete this->solver_;
    cublasDestroy(this->handle_);
}

void MainWindow::createSolver() {
    // load mesh
    auto nodes = fastEIT::matrix::loadtxt<fastEIT::dtype::real>("nodes.txt", NULL);
    auto elements = fastEIT::matrix::loadtxt<fastEIT::dtype::index>("elements.txt", NULL);
    auto boundary = fastEIT::matrix::loadtxt<fastEIT::dtype::index>("boundary.txt", NULL);

    // create mesh and electrodes
    auto mesh = new fastEIT::Mesh<fastEIT::basis::Linear>(*nodes, *elements, *boundary, 0.045, 0.1, NULL);
    auto electrodes = new fastEIT::Electrodes(36, 0.003, 0.003, 0.045);

    // create pattern
    auto drive_pattern = createPattern(36, 35, 2, NULL);
    auto measurement_pattern = createPattern(36, 0, 4, NULL);

    // create solver
    this->solver_ = new fastEIT::Solver<fastEIT::basis::Linear>(
                mesh, electrodes, *measurement_pattern, *drive_pattern,
                50e-3, 4, 0.05, this->handle(), NULL);

    // pre solve
    this->solver().preSolve(this->handle(), NULL);

    // cleanup
    delete nodes;
    delete elements;
    delete boundary;
    delete drive_pattern;
    delete measurement_pattern;
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
    // copy voltage
    if (this->measurement_system().isConnected()) {
        this->measurement_system().voltage().copyToDevice(NULL);
        this->solver().measured_voltage().copy(this->measurement_system().voltage(), NULL);
    }

    // solve
    this->time().restart();
    fastEIT::Matrix<fastEIT::dtype::real>& gamma = this->solver().solve(this->handle(), NULL);
    gamma.copyToHost(NULL);
    cudaStreamSynchronize(NULL);

    // calc fps
    this->fps_label().setText(QString("fps: %1").arg(1e3 / this->time().elapsed()));

    // cut values
    for (fastEIT::dtype::index element = 0; element < gamma.rows(); ++element) {
        if ((!this->ui->actionShow_Negative_Values->isChecked()) && (gamma(element, 0) < 0.0)) {
            gamma(element, 0) = 0.0;
        }
        if ((!this->ui->actionShow_Positive_Values->isChecked()) && (gamma(element, 0) > 0.0)) {
            gamma(element, 0) = 0.0;
        }
    }

    // get image
    Image* image = static_cast<Image*>(this->centralWidget());

    // update image
    fastEIT::dtype::real min_value, max_value;
    std::tie(min_value, max_value) = image->draw(gamma,
        this->ui->actionShow_Transparent_Values->isChecked());

    // update min max label
    this->min_label().setText(QString("min: %1 dB").arg(min_value));
    this->max_label().setText(QString("max: %1 dB").arg(max_value));
}

void MainWindow::measurementSystemConnectionError(QAbstractSocket::SocketError socket_error) {
    // check socket error
    if (socket_error == QAbstractSocket::HostNotFoundError) {
         QMessageBox::information(this, this->windowTitle(),
                                  tr("The host was not found. Please check the "
                                     "host name and port settings."));

    } else if (socket_error == QAbstractSocket::ConnectionRefusedError) {
         QMessageBox::information(this, this->windowTitle(),
                                  tr("The connection was refused by the measurement system. "
                                     "Make sure the system is running, "
                                     "and check that the host name and port "
                                     "settings are correct."));
    } else if (socket_error == QAbstractSocket::RemoteHostClosedError) {
         QMessageBox::information(this, this->windowTitle(),
                                  tr("The measurement system closed the connection. "
                                     "Make sure the system is still running."));
    }
}

void MainWindow::on_actionLoad_Voltage_triggered() {
    // get load file name
    QString file_name = QFileDialog::getOpenFileName(this, "Load Voltage",
                                                        "", "Matrix File (*.txt)");

    if (file_name != "") {
        // load matrix
        auto voltage = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(file_name.toStdString(), NULL);

        // copy voltage
        this->solver().measured_voltage().copy(*voltage, NULL);

        delete voltage;
    }
}

void MainWindow::on_actionSave_Voltage_triggered() {
    // get save file name
    QString file_name = QFileDialog::getSaveFileName(this, "Save Voltage", "",
                                                     "Matrix File (*.txt)");

    // save voltage
    if (file_name != "") {
        this->solver().measured_voltage().copyToHost(NULL);
        cudaStreamSynchronize(NULL);
        fastEIT::matrix::savetxt(file_name.toStdString(), this->solver().measured_voltage());
    }
}

void MainWindow::on_actionStart_Solver_triggered() {
    // start timer
    this->draw_timer().start(40);
}

void MainWindow::on_actionStop_Solver_triggered() {
    // stop timer
    this->draw_timer().stop();
}

void MainWindow::on_actionCalibrate_triggered() {
    // set calibration voltage to current measurment voltage
    this->solver().calibration_voltage().copy(this->solver().measured_voltage(), NULL);
}

void MainWindow::on_actionSave_Image_triggered() {
    // get image
    Image* image = static_cast<Image*>(this->centralWidget());

    // grap frame buffer
    QImage bitmap = image->grabFrameBuffer();

    // get save file name
    QString file_name = QFileDialog::getSaveFileName(this, "Save Image", "",
                                                    "PNG File (*.png)");

    // save image
    if (file_name != "") {
        bitmap.save(file_name, "PNG");
    }
}

void MainWindow::on_actionConnect_triggered() {
    // get host address
    bool ok;
    QString host_address = QInputDialog::getText(this, "Measurement System Host Address", "Host Address", QLineEdit::Normal, "127.0.0.1", &ok);

    // connect to system
    if (ok) {
        this->measurement_system().connectToSystem(QHostAddress(host_address), 3000);
    }
}

void MainWindow::on_actionDisconnect_triggered() {
   // disconnect from measurement system
    this->measurement_system().disconnectFromSystem();
}
