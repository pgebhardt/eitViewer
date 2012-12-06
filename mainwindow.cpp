#include <fasteit/fasteit.h>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) {
    ui->setupUi(this);

    // load mesh
    auto nodes = fastEIT::matrix::loadtxt<fastEIT::dtype::real>("nodes.txt", NULL);
    auto elements = fastEIT::matrix::loadtxt<fastEIT::dtype::index>("elements.txt", NULL);
    auto boundary = fastEIT::matrix::loadtxt<fastEIT::dtype::index>("boundary.txt", NULL);

    // create mesh and electrodes
    auto mesh = new fastEIT::Mesh<fastEIT::basis::Linear>(*nodes, *elements, *boundary, 0.045, 0.01, NULL);
    auto electrodes = new fastEIT::Electrodes(36, 0.003, 0.003, 0.045);

    // create image
    Image* image = new Image(*mesh, *electrodes);
    this->setCentralWidget(image);

    // test
    auto gamma = fastEIT::matrix::loadtxt<fastEIT::dtype::real>("gamma.txt", NULL);
    image->draw(*gamma, false);

    // cleanup
    delete nodes;
    delete elements;
    delete boundary;
    delete gamma;
}

MainWindow::~MainWindow()
{
    delete ui;
}
