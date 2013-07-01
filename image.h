#ifndef IMAGE_H
#define IMAGE_H

#include <QGLWidget>
#include <QtOpenGL>
#include <mpflow/mpflow.h>

class Image : public QGLWidget
{
    Q_OBJECT
public:
    explicit Image(QWidget* parent=nullptr);
    virtual ~Image();

    void init(std::shared_ptr<mpFlow::EIT::model::Base> model);
    void cleanup();
    void draw(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> values,
        bool normalized);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();
    virtual void mousePressEvent(QMouseEvent* event);
    virtual void mouseMoveEvent(QMouseEvent* event);
    virtual void wheelEvent(QWheelEvent* event);

public:
    // accessors
    std::shared_ptr<mpFlow::EIT::model::Base> model() {
        return this->model_;
    }
    std::vector<mpFlow::dtype::real>& red() { return this->red_; }
    std::vector<mpFlow::dtype::real>& green() { return this->green_; }
    std::vector<mpFlow::dtype::real>& blue() { return this->blue_; }
    std::vector<mpFlow::dtype::real>& node_area() { return this->node_area_; }
    std::vector<mpFlow::dtype::real>& element_area() { return this->element_area_; }
    mpFlow::dtype::real& x_angle() { return this->x_angle_; }
    mpFlow::dtype::real& z_angle() { return this->z_angle_; }
    std::tuple<int, int>& old_mouse_pos() { return this->old_mouse_pos_; }
    mpFlow::dtype::real& normalization_factor() { return this->normalization_factor_; }

private:
    std::shared_ptr<mpFlow::EIT::model::Base> model_;
    GLfloat* vertices_;
    GLfloat* colors_;
    std::vector<mpFlow::dtype::real> red_;
    std::vector<mpFlow::dtype::real> green_;
    std::vector<mpFlow::dtype::real> blue_;
    std::vector<mpFlow::dtype::real> node_area_;
    std::vector<mpFlow::dtype::real> element_area_;
    mpFlow::dtype::real x_angle_;
    mpFlow::dtype::real z_angle_;
    std::tuple<int, int> old_mouse_pos_;
    mpFlow::dtype::real normalization_factor_;
};

#endif // IMAGE_H
