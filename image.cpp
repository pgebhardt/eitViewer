#include <algorithm>
#include <cmath>
#include "image.h"

void jet(const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values, mpFlow::dtype::real norm,
         std::vector<mpFlow::dtype::real>* red, std::vector<mpFlow::dtype::real>* green,
         std::vector<mpFlow::dtype::real>* blue) {
    // calc colors
    for (mpFlow::dtype::index element = 0; element < values->rows(); ++element) {
        (*red)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm - 0.5) + 1.5,
                                            0.0), 1.0);
        (*green)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm - 0.0) + 1.5,
                                            0.0), 1.0);
        (*blue)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm + 0.5) + 1.5,
                                             0.0), 1.0);
    }
}

Image::Image(QWidget* parent) :
    QGLWidget(parent), model_(nullptr), vertices_(nullptr), colors_(nullptr),
    red_(0), green_(0), blue_(0), node_area_(0), element_area_(0), x_angle_(0.0),
    z_angle_(0.0), threashold_(0.1) {
}

Image::~Image() {
    this->cleanup();
}

void Image::init(std::shared_ptr<mpFlow::EIT::model::Base> model) {
    this->model_ = model;

    // cleanup
    this->cleanup();

    // create vectors
    this->red_ = std::vector<mpFlow::dtype::real>(this->model()->mesh()->elements()->rows(), 1.0);
    this->green_ = std::vector<mpFlow::dtype::real>(this->model()->mesh()->elements()->rows(), 1.0);
    this->blue_ = std::vector<mpFlow::dtype::real>(this->model()->mesh()->elements()->rows(), 1.0);
    this->node_area_ = std::vector<mpFlow::dtype::real>(this->model()->mesh()->nodes()->rows(), 0.0);
    this->element_area_ = std::vector<mpFlow::dtype::real>(this->model()->mesh()->elements()->rows(), 0.0);

    // create OpenGL buffer
    this->vertices_ = new GLfloat[model->mesh()->elements()->rows() * 3 * 3];
    this->colors_ = new GLfloat[model->mesh()->elements()->rows() * 3 * 4];

    // init buffer
    std::fill_n(this->vertices_, model->mesh()->elements()->rows() * 3 * 3, 0.0);
    std::fill_n(this->colors_, model->mesh()->elements()->rows() * 3 * 4, 1.0);

    // calc node and element area
    for (mpFlow::dtype::index element = 0; element < model->mesh()->elements()->rows(); ++element) {
        auto points = model->mesh()->elementNodes(element);

        this->element_area()[element] = 0.5 * std::abs(
            (std::get<0>(std::get<1>(points[1])) - std::get<0>(std::get<1>(points[0]))) *
            (std::get<1>(std::get<1>(points[2])) - std::get<1>(std::get<1>(points[0]))) -
            (std::get<0>(std::get<1>(points[2])) - std::get<0>(std::get<1>(points[0]))) *
            (std::get<1>(std::get<1>(points[1])) - std::get<1>(std::get<1>(points[0])))
            );

        for (mpFlow::dtype::index node = 0; node < 3; ++node) {
            this->node_area()[std::get<0>(points[node])] += 0.5 * std::abs(
                (std::get<0>(std::get<1>(points[1])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[2])) - std::get<1>(std::get<1>(points[0]))) -
                (std::get<0>(std::get<1>(points[2])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[1])) - std::get<1>(std::get<1>(points[0])))
                );
        }
    }

    // fill vertex buffer
    for (mpFlow::dtype::index element = 0; element < model->mesh()->elements()->rows(); ++element) {
        // get element nodes
        auto nodes = model->mesh()->elementNodes(element);

        for (mpFlow::dtype::index node = 0; node < 3; ++node) {
            this->vertices_[element * 3 * 3 + node * 3 + 0] =
                std::get<0>(std::get<1>(nodes[node])) / model->mesh()->radius();
            this->vertices_[element * 3 * 3 + node * 3 + 1] =
                std::get<1>(std::get<1>(nodes[node])) / model->mesh()->radius();
        }
    }
}

void Image::cleanup() {
    if (this->vertices_) {
        delete [] this->vertices_;
        this->vertices_ = nullptr;
    }
    if (this->colors_) {
        delete [] this->colors_;
        this->colors_ = nullptr;
    }
}

void Image::draw(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values) {
    // check for beeing initialized
    if ((!this->vertices_) || (!this->colors_) || (!this->model())) {
        return;
    }

    // min and max values
    mpFlow::dtype::real min_value = 0.0;
    mpFlow::dtype::real max_value = 0.0;

    // calc min and max
    for (mpFlow::dtype::index element = 0; element < values->rows(); ++element) {
        min_value = std::min((*values)(element, 0), min_value);
        max_value = std::max((*values)(element, 0), max_value);
    }

    // calc norm
    mpFlow::dtype::real norm = std::max(std::max(-min_value, max_value), this->threashold());

    // check norm to prevent division by zero
    norm = norm == 0.0 ? 1.0 : norm;

    // calc colors
    jet(values, norm, &this->red(), &this->green(), &this->blue());

    // set colors
    for (mpFlow::dtype::index element = 0; element < this->model()->mesh()->elements()->rows(); ++element) {
        // set red
        this->colors_[element * 3 * 4 + 0 * 4 + 0] =
        this->colors_[element * 3 * 4 + 1 * 4 + 0] =
        this->colors_[element * 3 * 4 + 2 * 4 + 0] =
            this->red()[element];

        // set green
        this->colors_[element * 3 * 4 + 0 * 4 + 1] =
        this->colors_[element * 3 * 4 + 1 * 4 + 1] =
        this->colors_[element * 3 * 4 + 2 * 4 + 1] =
            this->green()[element];

        // set blue
        this->colors_[element * 3 * 4 + 0 * 4 + 2] =
        this->colors_[element * 3 * 4 + 1 * 4 + 2] =
        this->colors_[element * 3 * 4 + 2 * 4 + 2] =
            this->blue()[element];
    }

    // calc z_values
    std::vector<mpFlow::dtype::real> z_values(this->model()->mesh()->nodes()->rows());
    for (mpFlow::dtype::index element = 0; element < this->model()->mesh()->elements()->rows(); ++element) {
        for (mpFlow::dtype::index node = 0; node < 3; ++node) {
            z_values[(*this->model()->mesh()->elements())(element, node)] +=
                (*values)(element, 0) * this->element_area()[element];
        }
    }

    // fill vertex buffer
    for (mpFlow::dtype::index element = 0; element < this->model()->mesh()->elements()->rows(); ++element) {
        for (mpFlow::dtype::index node = 0; node < 3; ++node) {
            this->vertices_[element * 3 * 3 + node * 3 + 2] =
                -z_values[(*this->model()->mesh()->elements())(element, node)] /
                    (this->node_area()[(*this->model()->mesh()->elements())(element, node)] * norm);
        }
    }

    // redraw
    this->updateGL();
}

void Image::initializeGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth (3.0);
}

void Image::resizeGL(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
}

void Image::paintGL() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glRotatef(this->x_angle(), 1.0, 0.0, 0.0);
    glRotatef(this->z_angle(), 0.0, 0.0, 1.0);

    // clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // check for beeing initialized
    if ((!this->vertices_) || (!this->colors_) || (!this->model())) {
        return;
    }

    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // set pointer
    glVertexPointer(3, GL_FLOAT, 0, this->vertices_);
    glColorPointer(4, GL_FLOAT, 0, this->colors_);

    // draw elements
    glDrawArrays(GL_TRIANGLES, 0, this->model()->mesh()->elements()->rows() * 3);

    // dactivate vertex arrays after drawing
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    // draw electrodes
    glBegin(GL_LINES);

    // first electrode will be marked red
    glColor3f(1.0, 0.0, 0.0);
    glVertex2f(std::get<0>(std::get<0>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(),
               std::get<1>(std::get<0>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius());
    glVertex2f(std::get<0>(std::get<1>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(),
               std::get<1>(std::get<1>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius());

    glLineWidth(5.0);
    glColor3f(0.0, 0.0, 0.0);
    for (mpFlow::dtype::index electrode = 1; electrode < this->model()->electrodes()->count(); ++electrode) {
        glVertex2f(std::get<0>(std::get<0>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(),
                   std::get<1>(std::get<0>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius());
        glVertex2f(std::get<0>(std::get<1>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(),
                   std::get<1>(std::get<1>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius());
    }
    glEnd();
}

void Image::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        this->old_mouse_pos() = std::make_tuple(event->x(), event->y());
    }
}

void Image::mouseMoveEvent(QMouseEvent *event) {
    if (event->buttons() == Qt::LeftButton) {
        this->z_angle() -= (std::get<0>(this->old_mouse_pos()) - event->x());
        this->x_angle() += (std::get<1>(this->old_mouse_pos()) - event->y());
        this->old_mouse_pos() = std::make_tuple(event->x(), event->y());
    }
}

void Image::wheelEvent(QWheelEvent* event) {
    this->threashold() += event->delta() > 0 ? 0.05 :
            this->threashold() >= 0.05 ? -0.05 : 0.0;
}
