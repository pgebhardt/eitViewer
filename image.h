#ifndef IMAGE_H
#define IMAGE_H

#include <QGLWidget>
#include <QtOpenGL>
#include <QTimer>
#include <mpflow/mpflow.h>

class Image : public QGLWidget {
    Q_OBJECT
public:
    explicit Image(QWidget* parent=nullptr);
    virtual ~Image();

    void init(std::shared_ptr<mpFlow::EIT::model::Base> model,
        mpFlow::dtype::index rows, mpFlow::dtype::index columns);
    void cleanup();

public slots:
    void reset_view();
    void update_data(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> data,
        double time_elapsed);
    void update_gl_buffer();

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();
    virtual void mousePressEvent(QMouseEvent* event);
    virtual void mouseMoveEvent(QMouseEvent* event);
    virtual void wheelEvent(QWheelEvent* event);

public:
    // accessors
    std::shared_ptr<mpFlow::EIT::model::Base> model() { return this->model_; }
    Eigen::ArrayXXf& data() { return this->data_; }
    Eigen::ArrayXXf& colors() { return this->colors_; }
    Eigen::ArrayXf& node_area() { return this->node_area_; }
    Eigen::ArrayXf& element_area() { return this->element_area_; }
    GLfloat* gl_vertices() { return this->gl_vertices_; }
    GLfloat* gl_colors() { return this->gl_colors_; }
    std::array<mpFlow::dtype::real, 2>& view_angle() { return this->view_angle_; }
    std::tuple<int, int>& old_mouse_pos() { return this->old_mouse_pos_; }
    mpFlow::dtype::real& threashold() { return this->threashold_; }
    QTimer& draw_timer() { return *this->draw_timer_; }
    double& image_pos() { return this->image_pos_; }
    double& image_increment() { return this->image_increment_; }

private:
    std::shared_ptr<mpFlow::EIT::model::Base> model_;
    Eigen::ArrayXXf data_;
    Eigen::ArrayXXf colors_;
    Eigen::ArrayXf node_area_;
    Eigen::ArrayXf element_area_;
    GLfloat* gl_vertices_;
    GLfloat* gl_colors_;
    std::array<mpFlow::dtype::real, 2> view_angle_;
    std::tuple<int, int> old_mouse_pos_;
    mpFlow::dtype::real threashold_;
    QTimer* draw_timer_;
    double image_pos_;
    double image_increment_;
};

#endif // IMAGE_H
