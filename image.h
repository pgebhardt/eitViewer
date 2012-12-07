#ifndef IMAGE_H
#define IMAGE_H

#include <QGLWidget>
#include <QtOpenGL>
#include <fasteit/fasteit.h>

class Image : public QGLWidget
{
    Q_OBJECT
public:
    explicit Image(const fastEIT::Mesh<fastEIT::basis::Linear>& mesh,
                   const fastEIT::Electrodes& electrodes,
                   QWidget *parent = 0);
    virtual ~Image();

signals:

public slots:

public:
    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> draw(
        fastEIT::Matrix<fastEIT::dtype::real>& values, bool transparent);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

public:
    // accessors
    const fastEIT::Mesh<fastEIT::basis::Linear>& mesh() const {
        return this->mesh_;
    }
    const fastEIT::Electrodes& electrodes() const {
        return this->electrodes_;
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

private:
    const fastEIT::Mesh<fastEIT::basis::Linear>& mesh_;
    const fastEIT::Electrodes& electrodes_;
    GLfloat* vertices_;
    GLfloat* colors_;
    std::vector<fastEIT::dtype::real> red_;
    std::vector<fastEIT::dtype::real> green_;
    std::vector<fastEIT::dtype::real> blue_;
};

#endif // IMAGE_H
