#include "image.h"

Image::Image(const fastEIT::Mesh<fastEIT::basis::Linear>& mesh,
             const fastEIT::Electrodes& electrodes,
             QWidget *parent) :
    QGLWidget(parent)
{
}
