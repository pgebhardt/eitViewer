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
    void draw(const fastEIT::Matrix<fastEIT::dtype::real>& values,
              bool transparent);

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
    const fastEIT::dtype::real& min_value() const {
        return this->min_value_;
    }
    const fastEIT::dtype::real& max_value() const {
        return this->max_value_;
    }

    // mutators
    fastEIT::dtype::real& min_value() { return this->min_value_; }
    fastEIT::dtype::real& max_value() { return this->max_value_; }

private:
    const fastEIT::Mesh<fastEIT::basis::Linear>& mesh_;
    const fastEIT::Electrodes& electrodes_;
    GLfloat* vertices_;
    GLfloat* colors_;
    fastEIT::dtype::real min_value_;
    fastEIT::dtype::real max_value_;
};

#endif // IMAGE_H
