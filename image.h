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
    
signals:
    
public slots:

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

protected:
    // accessors
    const fastEIT::Mesh<fastEIT::basis::Linear>& mesh() const {
        return this->mesh_;
    }
    const fastEIT::Electrodes& electrodes() const {
        return this->electrodes_;
    }

private:
    const fastEIT::Mesh<fastEIT::basis::Linear>& mesh_;
    const fastEIT::Electrodes& electrodes_;
};

#endif // IMAGE_H
