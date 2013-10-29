#include <algorithm>
#include <cmath>
#include "image.h"

Image::Image(QWidget* parent) :
    QGLWidget(parent), threashold_(0.1), image_pos_(0.0), image_increment_(0.0),
    sigma_ref_(0.0) {
    // create timer
    this->draw_timer_ = new QTimer(this);
    connect(this->draw_timer_, &QTimer::timeout, this, &Image::update_gl_buffer);

    // init view
    this->reset_view();
}

Image::~Image() {
    this->cleanup();
}

void Image::init(std::shared_ptr<mpFlow::EIT::model::Base> model,
    mpFlow::dtype::index rows, mpFlow::dtype::index columns) {
    // cleanup
    this->cleanup();

    // create arrays
    this->data() = Eigen::ArrayXXf::Ones(rows, columns) * model->sigma_ref();
    this->vertices() = Eigen::ArrayXXf::Zero(3 * 3, model->mesh()->elements()->rows());
    this->colors() = Eigen::ArrayXXf::Zero(3 * 3, model->mesh()->elements()->rows());
    this->electrodes() = Eigen::ArrayXXf::Zero(2 * 2, model->electrodes()->count());
    this->electrode_colors() = Eigen::ArrayXXf::Zero(3 * 2, model->electrodes()->count());
    this->elements() = mpFlow::numeric::matrix::toEigen<mpFlow::dtype::index>(
        model->mesh()->elements());
    this->z_values() = Eigen::ArrayXf::Zero(model->mesh()->nodes()->rows());
    this->element_area() = Eigen::ArrayXf::Zero(model->mesh()->elements()->rows());
    this->node_area() = Eigen::ArrayXf::Zero(model->mesh()->nodes()->rows());

    // calc node and element area
    Eigen::ArrayXXf nodes = mpFlow::numeric::matrix::toEigen<mpFlow::dtype::real>(model->mesh()->nodes());
    for (mpFlow::dtype::index element = 0; element < this->elements().rows(); ++element) {
        this->element_area()(element) = 0.5 * std::abs(
            (nodes(this->elements()(element, 1), 0) - nodes(this->elements()(element, 0), 0)) *
            (nodes(this->elements()(element, 2), 1) - nodes(this->elements()(element, 0), 1)) -
            (nodes(this->elements()(element, 2), 0) - nodes(this->elements()(element, 0), 0)) *
            (nodes(this->elements()(element, 1), 1) - nodes(this->elements()(element, 0), 1)));

        for (mpFlow::dtype::index node = 0; node < 3; ++node) {
            this->node_area()(this->elements()(element, node)) += this->element_area()(element);
        }
    }

    // fill vertex buffer
    for (mpFlow::dtype::index element = 0; element < this->elements().rows(); ++element)
    for (mpFlow::dtype::index node = 0; node < 3; ++node) {
        this->vertices()(node * 3 + 0, element) =
            nodes(this->elements()(element, node), 0) / model->mesh()->radius();
        this->vertices()(node * 3 + 1, element) =
            nodes(this->elements()(element, node), 1) / model->mesh()->radius();
    }

    // fill electrodes buffer
    for (mpFlow::dtype::index electrode = 0; electrode < model->electrodes()->count(); ++electrode) {
        this->electrodes()(0 * 2 + 0, electrode) = std::get<0>(std::get<0>(
            model->electrodes()->coordinates(electrode))) / model->mesh()->radius();
        this->electrodes()(0 * 2 + 1, electrode) = std::get<1>(std::get<0>(
            model->electrodes()->coordinates(electrode))) / model->mesh()->radius();
        this->electrodes()(1 * 2 + 0, electrode) = std::get<0>(std::get<1>(
            model->electrodes()->coordinates(electrode))) / model->mesh()->radius();
        this->electrodes()(1 * 2 + 1, electrode) = std::get<1>(std::get<1>(
            model->electrodes()->coordinates(electrode))) / model->mesh()->radius();
    }

    // mark first electrode red
    this->electrode_colors()(0, 0) = this->electrode_colors()(3, 0) = 1.0;

    // save sigma ref
    this->sigma_ref() = model->sigma_ref();

    // redraw
    this->update_gl_buffer();
    this->updateGL();
}

void Image::cleanup() {
    // stop timer
    this->draw_timer().stop();

    // reset view
    this->reset_view();

    // reset image buffer pos and increment
    this->image_pos() = 0.0;
    this->image_increment() = 0.0;

    // redraw
    this->updateGL();
}

void Image::reset_view() {
    this->view_angle()[0] = 0.0;
    this->view_angle()[1] = 0.0;
    this->threashold() = 0.05;
}

void Image::update_data(Eigen::ArrayXXf data, double time_elapsed) {
    this->data() = data;

    // update image increments
    this->image_pos() = 0.0;
    this->image_increment() = time_elapsed > 0.02 ? 0.02 / time_elapsed *
        (double)this->data().cols() : 0.0;

    // start draw timer
    this->draw_timer().start(20);
}

void Image::update_gl_buffer() {
    // calc norm and prevent division by zero and normalize data
    mpFlow::dtype::real norm = std::max(std::max(
        -(this->data().col(this->image_pos()) - this->sigma_ref()).minCoeff(),
        (this->data().col(this->image_pos()) - this->sigma_ref()).maxCoeff()),
        this->threashold() * this->sigma_ref());
    norm = norm == 0.0 ? 1.0 : norm;

    // subtract reference conductivity and normalize current data set
    auto normalized_data = (this->data().col(this->image_pos())
        - this->sigma_ref()) / norm;

    // calc colors
    this->colors().row(0) = this->colors().row(3) = this->colors().row(6) =
        (-2.0 * (normalized_data - 0.5).abs() + 1.5).max(0.0).min(1.0);
    this->colors().row(1) = this->colors().row(4) = this->colors().row(7) =
        (-2.0 * (normalized_data - 0.0).abs() + 1.5).max(0.0).min(1.0);
    this->colors().row(2) = this->colors().row(5) = this->colors().row(8) =
        (-2.0 * (normalized_data + 0.5).abs() + 1.5).max(0.0).min(1.0);

    // calc z values
    this->z_values().setZero();
    for (mpFlow::dtype::index element = 0; element < this->elements().rows(); ++element)
    for (mpFlow::dtype::index node = 0; node < 3; ++node) {
        this->z_values()(this->elements()(element, node)) -=
            normalized_data(element) * this->element_area()(element) /
            this->node_area()(this->elements()(element, node));
    }

    // copy z values to opengl vertex buffer
    for (mpFlow::dtype::index element = 0; element < this->elements().rows(); ++element)
    for (mpFlow::dtype::index node = 0; node < 3; ++node) {
        this->vertices()(node * 3 + 2, element) =
            this->z_values()(this->elements()(element, node));
    }

    // update image pos
    this->image_pos() += this->image_increment();
    if (this->image_pos() >= (double)this->data().cols()) {
        this->image_pos() = (double)this->data().cols() - 1.0;

        // stop draw timer
        this->draw_timer().stop();
    }

    // redraw
    this->updateGL();
}

void Image::initializeGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(3.0);
}

void Image::resizeGL(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glViewport(
        0.5 * w - 0.5 * std::min(w, h),
        0.5 * h - 0.5 * std::min(w, h),
        std::min(w, h), std::min(w, h));
    glMatrixMode(GL_MODELVIEW);
}

void Image::paintGL() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glRotatef(this->view_angle()[0], 1.0, 0.0, 0.0);
    glRotatef(this->view_angle()[1], 0.0, 0.0, 1.0);

    // clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // draw mesh
    glVertexPointer(3, GL_FLOAT, 0, this->vertices().data());
    glColorPointer(3, GL_FLOAT, 0, this->colors().data());
    glDrawArrays(GL_TRIANGLES, 0, this->vertices().cols() * 3);

    // draw electrodes
    glVertexPointer(2, GL_FLOAT, 0, this->electrodes().data());
    glColorPointer(3, GL_FLOAT, 0, this->electrode_colors().data());
    glDrawArrays(GL_LINES, 0, this->electrodes().cols() * 2);

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
        this->view_angle()[1] -= (std::get<0>(this->old_mouse_pos()) - event->x());
        this->view_angle()[0] += (std::get<1>(this->old_mouse_pos()) - event->y());
        this->old_mouse_pos() = std::make_tuple(event->x(), event->y());

        if (!this->draw_timer().isActive()) {
            this->updateGL();
        }
    }
}

void Image::wheelEvent(QWheelEvent* event) {
    this->threashold() += event->delta() > 0 ? 0.01 :
            this->threashold() >= 0.01 ? -0.01 : -this->threashold();

    if (!this->draw_timer().isActive()) {
        this->update_gl_buffer();
        this->updateGL();
    }
}
