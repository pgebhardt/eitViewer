#ifndef IMAGE_H
#define IMAGE_H

#include <QGLWidget>
#include <QtOpenGL>
#include <fasteit/fasteit.h>

class Image : public QGLWidget
{
    Q_OBJECT
public:
    explicit Image(QWidget* parent=nullptr);
    virtual ~Image();

    void init(std::shared_ptr<fastEIT::model::Model> model);
    void cleanup();
    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> draw(
        const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> values, bool normalized);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();
    virtual void mousePressEvent(QMouseEvent* event);
    virtual void mouseMoveEvent(QMouseEvent* event);
    virtual void wheelEvent(QWheelEvent* event);

public:
    // accessors
    std::shared_ptr<fastEIT::model::Model> model() const {
        return this->model_;
    }
    const std::vector<fastEIT::dtype::real>& red() const {
        return this->red_;
    }
    const std::vector<fastEIT::dtype::real>& green() const {
        return this->green_;
    }
    const std::vector<fastEIT::dtype::real>& blue() const {
        return this->blue_;
    }

    // mutators
    std::vector<fastEIT::dtype::real>& red() { return this->red_; }
    std::vector<fastEIT::dtype::real>& green() { return this->green_; }
    std::vector<fastEIT::dtype::real>& blue() { return this->blue_; }
    std::vector<fastEIT::dtype::real>& node_area() { return this->node_area_; }
    std::vector<fastEIT::dtype::real>& element_area() { return this->element_area_; }
    fastEIT::dtype::real& x_angle() { return this->x_angle_; }
    fastEIT::dtype::real& z_angle() { return this->z_angle_; }
    std::tuple<int, int>& old_mouse_pos() { return this->old_mouse_pos_; }
    fastEIT::dtype::real& normalization_factor() { return this->normalization_factor_; }

private:
    std::shared_ptr<fastEIT::model::Model> model_;
    GLfloat* vertices_;
    GLfloat* colors_;
    std::vector<fastEIT::dtype::real> red_;
    std::vector<fastEIT::dtype::real> green_;
    std::vector<fastEIT::dtype::real> blue_;
    std::vector<fastEIT::dtype::real> node_area_;
    std::vector<fastEIT::dtype::real> element_area_;
    fastEIT::dtype::real x_angle_;
    fastEIT::dtype::real z_angle_;
    std::tuple<int, int> old_mouse_pos_;
    fastEIT::dtype::real normalization_factor_;
};

#endif // IMAGE_H
