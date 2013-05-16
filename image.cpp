#include <algorithm>
#include <cmath>
#include "image.h"

void jet(const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> values, fastEIT::dtype::real norm,
         std::vector<fastEIT::dtype::real>* red, std::vector<fastEIT::dtype::real>* green,
         std::vector<fastEIT::dtype::real>* blue) {
    // calc colors
    for (fastEIT::dtype::index element = 0; element < values->rows(); ++element) {
        (*red)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm - 0.5) + 1.5,
                                            0.0), 1.0);
        (*green)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm - 0.0) + 1.5,
                                            0.0), 1.0);
        (*blue)[element] = std::min(std::max(-2.0 * std::abs((*values)(element, 0) / norm + 0.5) + 1.5,
                                             0.0), 1.0);
    }
}

void calc_node_area(const std::shared_ptr<fastEIT::Mesh> mesh,
              std::vector<fastEIT::dtype::real>* area) {
    for (fastEIT::dtype::index element = 0; element < mesh->elements()->rows(); ++element) {
        auto points = mesh->elementNodes(element);

        for (fastEIT::dtype::index node = 0; node < 3; ++node) {
            (*area)[std::get<0>(points[node])] += 0.5 * std::abs(
                (std::get<0>(std::get<1>(points[1])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[2])) - std::get<1>(std::get<1>(points[0]))) -
                (std::get<0>(std::get<1>(points[2])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[1])) - std::get<1>(std::get<1>(points[0])))
                );
        }
    }
}

void calc_z_values(const std::vector<fastEIT::dtype::real>& node_area,
                  const std::shared_ptr<fastEIT::Mesh> mesh,
                  const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> values,
                  const fastEIT::dtype::real norm, GLfloat* vertices) {
    // calc z_values
    std::vector<fastEIT::dtype::real> z_values(mesh->nodes()->rows());
    for (fastEIT::dtype::index element = 0; element < mesh->elements()->rows(); ++element) {
        // get element nodes
        auto points = mesh->elementNodes(element);

        for (fastEIT::dtype::index node = 0; node < 3; ++node) {
            z_values[std::get<0>(points[node])] += (*values)(element, 0) * 0.5 * std::abs(
                (std::get<0>(std::get<1>(points[1])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[2])) - std::get<1>(std::get<1>(points[0]))) -
                (std::get<0>(std::get<1>(points[2])) - std::get<0>(std::get<1>(points[0]))) *
                (std::get<1>(std::get<1>(points[1])) - std::get<1>(std::get<1>(points[0])))
                );
        }
    }

    // fill vertex buffer
    for (fastEIT::dtype::index element = 0; element < mesh->elements()->rows(); ++element) {
        // get element nodes
        auto points = mesh->elementNodes(element);

        for (fastEIT::dtype::index node = 0; node < 3; ++node) {
            vertices[element * 3 * 3 + node * 3 + 2] = -z_values[std::get<0>(points[node])] / (node_area[std::get<0>(points[node])] * norm);
        }
    }
}

Image::Image(const std::shared_ptr<fastEIT::model::Model> model, QWidget *parent) :
    QGLWidget(parent), model_(model), red_(model->mesh()->elements()->rows()),
    green_(model->mesh()->elements()->rows()), blue_(model->mesh()->elements()->rows()),
    node_area_(model->mesh()->elements()->rows()), x_angle_(0.0), z_angle_(0.0),
    normalization_factor_(1.0) {

    // create buffer
    this->vertices_ = new GLfloat[model->mesh()->elements()->rows() * 3 * 3];
    this->colors_ = new GLfloat[model->mesh()->elements()->rows() * 3 * 4];

    // init buffer
    std::fill_n(this->vertices_, model->mesh()->elements()->rows() * 3 * 3, 0.0);
    std::fill_n(this->colors_, model->mesh()->elements()->rows() * 3 * 4, 1.0);

    // calc node area
    calc_node_area(model->mesh(), &this->node_area());

    // fill vertex buffer
    for (fastEIT::dtype::index element = 0; element < model->mesh()->elements()->rows(); ++element) {
        // get element nodes
        auto nodes = model->mesh()->elementNodes(element);

        for (fastEIT::dtype::index node = 0; node < 3; ++node) {
            this->vertices_[element * 3 * 3 + node * 3 + 0] =
                std::get<0>(std::get<1>(nodes[node])) / model->mesh()->radius();
            this->vertices_[element * 3 * 3 + node * 3 + 1] =
                std::get<1>(std::get<1>(nodes[node])) / model->mesh()->radius();
        }
    }
}

Image::~Image() {
    delete this->vertices_;
    delete this->colors_;
}

std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> Image::draw(
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> values, bool transparent,
    bool normalized) {

    // min and max values
    fastEIT::dtype::real min_value = 0.0;
    fastEIT::dtype::real max_value = 0.0;

    // calc min and max
    for (fastEIT::dtype::index element = 0; element < values->rows(); ++element) {
        min_value = std::min((*values)(element, 0), min_value);
        max_value = std::max((*values)(element, 0), max_value);
    }

    // calc norm
    fastEIT::dtype::real norm = normalized ? std::max(-min_value, max_value) : this->normalization_factor();

    // check norm to prevent division by zero
    norm = norm == 0.0 ? 1.0 : norm;

    // calc colors
    jet(values, norm, &this->red(), &this->green(), &this->blue());

    // set colors
    for (fastEIT::dtype::index element = 0; element < this->model()->mesh()->elements()->rows(); ++element) {
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

        // calc alpha
        if (transparent) {
            this->colors_[element * 3 * 4 + 0 * 4 + 3] =
            this->colors_[element * 3 * 4 + 1 * 4 + 3] =
            this->colors_[element * 3 * 4 + 2 * 4 + 3] =
                std::abs((*values)(element, 0) / norm);
        } else {
            this->colors_[element * 3 * 4 + 0 * 4 + 3] =
            this->colors_[element * 3 * 4 + 1 * 4 + 3] =
            this->colors_[element * 3 * 4 + 2 * 4 + 3] =
                1.0;
        }
    }

    // calc z_values
    calc_z_values(this->node_area(), this->model()->mesh(), values, norm, this->vertices_);

    // redraw
    this->updateGL();

    return std::make_tuple(min_value, max_value);
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

    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // set pointer
    glVertexPointer(3, GL_FLOAT, 0, this->vertices_);
    glColorPointer(4, GL_FLOAT, 0, this->colors_);

    // draw elements
    glDrawArrays(GL_TRIANGLES, 0, this->model()->mesh()->elements()->rows() * 3);

    // draw electrodes
    glBegin(GL_LINES);

    // first electrode will be marked red
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(std::get<0>(std::get<0>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(),
               std::get<1>(std::get<0>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(), 0.0);
    glVertex3f(std::get<0>(std::get<1>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(),
               std::get<1>(std::get<1>(this->model()->electrodes()->coordinates(0))) / this->model()->mesh()->radius(), 0.0);

    glLineWidth(5.0);
    glColor3f(0.0, 0.0, 0.0);
    for (fastEIT::dtype::index electrode = 1; electrode < this->model()->electrodes()->count(); ++electrode) {
        glVertex3f(std::get<0>(std::get<0>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(),
                   std::get<1>(std::get<0>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(), 0.0);
        glVertex3f(std::get<0>(std::get<1>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(),
                   std::get<1>(std::get<1>(this->model()->electrodes()->coordinates(electrode))) / this->model()->mesh()->radius(), 0.0);
    }
    glEnd();

    // dactivate vertex arrays after drawing
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
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
    this->normalization_factor() *= event->delta() > 0 ? 2.0 : 0.5;
}
