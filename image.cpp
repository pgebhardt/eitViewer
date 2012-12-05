#include "image.h"

Image::Image(const fastEIT::Mesh<fastEIT::basis::Linear>& mesh,
             const fastEIT::Electrodes& electrodes,
             QWidget *parent) :
    QGLWidget(parent), mesh_(mesh), electrodes_(electrodes) {

}

void Image::initializeGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Image::resizeGL(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
}

void Image::paintGL() {
    // clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // TODO: set pointer
    // TODO: draw elements

    // draw electrodes
    glLineWidth(3.0);
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0);

    for (fastEIT::dtype::index electrode = 0; electrode < this->electrodes().count(); ++electrode) {
        glVertex3f(-std::get<1>(this->electrodes().electrodes_start()[electrode]) / this->mesh().radius(),
                   -std::get<0>(this->electrodes().electrodes_start()[electrode]) / this->mesh().radius(), 0.0);
        glVertex3f(-std::get<1>(this->electrodes().electrodes_end()[electrode]) / this->mesh().radius(),
                   -std::get<0>(this->electrodes().electrodes_end()[electrode]) / this->mesh().radius(), 0.0);
    }
    glEnd();

    // dactivate vertex arrays after drawing
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}
