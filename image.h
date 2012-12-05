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
    
};

#endif // IMAGE_H
