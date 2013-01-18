#ifndef IMAGE_H
#define IMAGE_H

#include <QGLWidget>
#include <QtOpenGL>
#include <fasteit/fasteit.h>

class Image : public QGLWidget
{
    Q_OBJECT
public:
    explicit Image(const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model, QWidget *parent = 0);
    virtual ~Image();

    void init(const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model);

signals:

public slots:

public:
    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> draw(
        const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> values, bool transparent);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

public:
    // accessors
    const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model() const {
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

private:
    const std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model_;
    GLfloat* vertices_;
    GLfloat* colors_;
    std::vector<fastEIT::dtype::real> red_;
    std::vector<fastEIT::dtype::real> green_;
    std::vector<fastEIT::dtype::real> blue_;
};

#endif // IMAGE_H
