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
    void update_data(Eigen::ArrayXXf data, double time_elapsed);
    void update_gl_buffer();
    void set_draw_wireframe(bool draw_wireframe);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();
    virtual void mousePressEvent(QMouseEvent* event);
    virtual void mouseMoveEvent(QMouseEvent* event);
    virtual void wheelEvent(QWheelEvent* event);

public:
    // accessors
    Eigen::ArrayXXf& data() { return this->data_; }
    Eigen::ArrayXXf& vertices() { return this->vertices_; }
    Eigen::ArrayXXf& colors() { return this->colors_; }
    Eigen::ArrayXXf& electrodes() { return this->electrodes_; }
    Eigen::ArrayXXf& electrode_colors() { return this->electrode_colors_; }
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic>& elements() { return this->elements_; }
    Eigen::ArrayXf& z_values() { return this->z_values_; }
    Eigen::ArrayXf& node_area() { return this->node_area_; }
    Eigen::ArrayXf& element_area() { return this->element_area_; }
    std::array<mpFlow::dtype::real, 2>& view_angle() { return this->view_angle_; }
    std::tuple<int, int>& old_mouse_pos() { return this->old_mouse_pos_; }
    mpFlow::dtype::real& threashold() { return this->threashold_; }
    QTimer& draw_timer() { return *this->draw_timer_; }
    double& image_pos() { return this->image_pos_; }
    double& image_increment() { return this->image_increment_; }
    mpFlow::dtype::real& sigma_ref() { return this->sigma_ref_; }
    bool draw_wireframe() { return this->draw_wireframe_; }

private:
    Eigen::ArrayXXf data_;
    Eigen::ArrayXXf vertices_;
    Eigen::ArrayXXf colors_;
    Eigen::ArrayXXf electrodes_;
    Eigen::ArrayXXf electrode_colors_;
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic> elements_;
    Eigen::ArrayXf z_values_;
    Eigen::ArrayXf node_area_;
    Eigen::ArrayXf element_area_;
    std::array<mpFlow::dtype::real, 2> view_angle_;
    std::tuple<int, int> old_mouse_pos_;
    mpFlow::dtype::real threashold_;
    QTimer* draw_timer_;
    double image_pos_;
    double image_increment_;
    mpFlow::dtype::real sigma_ref_;
    bool draw_wireframe_;
};

#endif // IMAGE_H
