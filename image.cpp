#include <algorithm>
#include "image.h"

Image::Image(const fastEIT::Mesh<fastEIT::basis::Linear>& mesh,
             const fastEIT::Electrodes& electrodes,
             QWidget *parent) :
    QGLWidget(parent), mesh_(mesh), electrodes_(electrodes), min_value_(0.0), max_value_(0.0) {
    // create buffer
    this->vertices_ = new GLfloat[mesh.elements().rows() * mesh.elements().columns() * 2];
    this->colors_ = new GLfloat[mesh.elements().rows() * mesh.elements().columns() * 4];

    // init buffer
    std::fill_n(this->vertices_, mesh.elements().rows() * mesh.elements().columns() * 2, 0.0);
    std::fill_n(this->colors_, mesh.elements().rows() * mesh.elements().columns() * 4, 1.0);

    // fill vertex buffer
    for (fastEIT::dtype::index element = 0; element < mesh.elements().rows(); ++element) {
        // get element nodes
        auto nodes = mesh.elementNodes(element);

        for (fastEIT::dtype::index node = 0; node < nodes.size(); ++node) {
            this->vertices_[element * mesh.elements().columns() * 2
                    + node * 2 + 0] = -std::get<1>(nodes[node]) / mesh.radius();
            this->vertices_[element * mesh.elements().columns() * 2
                    + node * 2 + 1] = -std::get<0>(nodes[node]) / mesh.radius();
        }
    }
}

Image::~Image() {
    delete this->vertices_;
    delete this->colors_;
}

void Image::draw(const fastEIT::Matrix<fastEIT::dtype::real> &values, bool transparent) {
    // reset min and max
    std::tie(this->min_value(), this->max_value()) = std::make_tuple(0.0, 0.0);

    // calc min and max
    for (fastEIT::dtype::index element = 0; element < values.rows(); ++element) {
        this->min_value() = std::min(values(element, 0), this->min_value());
        this->max_value() = std::max(values(element, 0), this->max_value());
    }

    // change color
    this->colors_[0] = this->colors_[1] = this->colors_[2] = 0.0;
    this->colors_[4] = this->colors_[5] = this->colors_[6] = 0.0;
    this->colors_[8] = this->colors_[9] = this->colors_[10] = 0.0;
}

void Image::initializeGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Image::resizeGL(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
}

void Image::paintGL() {
    // clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // set pointer
    glVertexPointer(2, GL_FLOAT, 0, this->vertices_);
    glColorPointer(4, GL_FLOAT, 0, this->colors_);

    // draw elements
    glDrawArrays(GL_TRIANGLES, 0, this->mesh_.elements().rows() * this->mesh_.elements().columns());

    // draw electrodes
    glLineWidth(3.0);
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0);

    for (fastEIT::dtype::index electrode = 0; electrode < this->electrodes().count(); ++electrode) {
        glVertex3f(-std::get<1>(this->electrodes().electrodes_start()[electrode]) / this->mesh().radius(),
                   -std::get<0>(this->electrodes().electrodes_start()[electrode]) / this->mesh().radius(), 0.0);
        glVertex3f(-std::get<1>(this->electrodes().electrodes_end()[electrode]) / this->mesh().radius(),
                   -std::get<0>(this->electrodes().electrodes_end()[electrode]) / this->mesh().radius(), 0.0);
    }
    glEnd();

    // dactivate vertex arrays after drawing
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}
