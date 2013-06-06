#ifndef FIRFILTER_H
#define FIRFILTER_H

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QTime>
#include <fasteit/fasteit.h>

class FIRFilter : public QObject {
    Q_OBJECT
public:
    explicit FIRFilter(unsigned int order, unsigned int step_size, int cuda_device,
        std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> input,
        QObject *parent=nullptr);
    void restart(int step_size);
    
signals:
    void initialized(bool success);

protected slots:
    virtual void calc_filter();

public:
    // accessors
    QThread* thread() { return this->thread_; }
    QTimer* timer() { return this->timer_; }
    cudaStream_t cuda_stream() { return this->cuda_stream_; }
    cublasHandle_t cublas_handle() { return this->cublas_handle_; }
    std::vector<std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>>& buffer() {
        return this->buffer_;
    }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> input() { return this->input_; }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> calc_array() { return this->calc_array_; }
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> output() { return this->output_; }
    std::vector<fastEIT::dtype::real>& filter_coefficients() {
        return this->filter_coefficients_;
    }
    unsigned int order() { return this->order_; }
    fastEIT::dtype::index& ring_buffer_pos() { return this->ring_buffer_pos_; }
    int step_size() { return this->step_size_; }

private:
    // member
    QThread* thread_;
    QTimer* timer_;
    cudaStream_t cuda_stream_;
    cublasHandle_t cublas_handle_;
    std::vector<std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>> buffer_;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> input_;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> calc_array_;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> output_;
    std::vector<fastEIT::dtype::real> filter_coefficients_;
    unsigned int order_;
    fastEIT::dtype::index ring_buffer_pos_;
    int step_size_;
    
};

#endif // FIRFILTER_H
