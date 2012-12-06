#include <fasteit/fasteit.h>

#include <QtCore>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"

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
    QTimer* draw_timer = new QTimer(this);
    connect(draw_timer, SIGNAL(timeout()), this, SLOT(draw()));

    // create cublas handle
    cublasCreate(&this->handle_);

    // create solver
    this->createSolver();

    draw_timer->start(40);
}

MainWindow::~MainWindow() {
    delete ui;
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

    // create image
    this->setCentralWidget(new Image(*mesh, *electrodes));

    // create pattern
    auto drive_pattern = createPattern(36, 35, 2, NULL);
    auto measurment_pattern = createPattern(36, 0, 4, NULL);

    // create solver
    this->solver_ = new fastEIT::Solver<fastEIT::basis::Linear>(
                mesh, electrodes, *measurment_pattern, *drive_pattern,
                50e-3, 4, 0.05, this->handle(), NULL);

    // pre solve
    this->solver().preSolve(this->handle(), NULL);

    // cleanup
    delete nodes;
    delete elements;
    delete boundary;
    delete drive_pattern;
    delete measurment_pattern;
}

void MainWindow::on_actionLoad_Voltage_triggered() {

}

void MainWindow::draw() {
    // solve
    auto gamma = this->solver().solve(this->handle(), NULL);
    gamma.copyToHost(NULL);

    // update image
    static_cast<Image*>(this->centralWidget())->draw(gamma, true);
}
