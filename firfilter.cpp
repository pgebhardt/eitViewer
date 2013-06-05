#include "firfilter.h"

FIRFilter::FIRFilter(unsigned int order, unsigned int time_step_size, int cuda_device,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> input, QObject* parent) :
    QObject(parent), thread_(nullptr), timer_(nullptr), cuda_stream_(nullptr),
    input_(input), calc_array_(nullptr), output_(nullptr), order_(order), ring_buffer_pos_(0) {
    // init separate thread for execution of filter
    this->thread_ = new QThread();
    this->moveToThread(this->thread());

    connect(this->thread(), &QThread::started, [=]() {
        // switch to cuda device and create cublas handle
        cudaSetDevice(cuda_device);
        cudaStreamCreate(&this->cuda_stream_);
        cublasCreate(&this->cublas_handle_);
        cublasSetStream(this->cublas_handle(), this->cuda_stream());

        // init buffer and filter response
        this->filter_coefficients() = std::vector<fastEIT::dtype::real>(
            this->order() + 1, 1.0 / (fastEIT::dtype::real)(this->order() + 1));

        for (fastEIT::dtype::index i = 0; i <= this->order(); ++i) {
            this->buffer().push_back(std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(
                this->input()->rows(), this->input()->columns(), this->cuda_stream()));
        }

        // create output matrix
        this->calc_array_ = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(
            this->input()->rows(), this->input()->columns(), this->cuda_stream());
        this->output_ = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(
            this->input()->rows(), this->input()->columns(), this->cuda_stream());

        // create timer for timing filter execution
        this->timer_ = new QTimer();
        connect(this->timer(), &QTimer::timeout, this, &FIRFilter::calc_filter);
        this->timer()->start(time_step_size);

        emit this->initialized(true);
    });
    this->thread()->start();
}

void FIRFilter::calc_filter() {
    // set new input to buffer
    this->buffer()[this->ring_buffer_pos()]->copy(this->input(), this->cuda_stream());

    // eval fir filter
    this->calc_array()->copy(this->buffer()[0], this->cuda_stream());

    this->calc_array()->scalarMultiply(this->filter_coefficients()[
        (-this->ring_buffer_pos()) % this->buffer().size()],
        this->cuda_stream());

    for (fastEIT::dtype::index pos = 1; pos < this->buffer().size(); ++pos) {
        cublasSaxpy(this->cublas_handle(), this->buffer()[pos]->data_rows() *
            this->buffer()[pos]->data_columns(),
            &this->filter_coefficients()[
                (pos - this->ring_buffer_pos()) % this->buffer().size()],
            this->buffer()[pos]->device_data(), 1, this->calc_array()->device_data(), 1);
    }

    // copy result to output array
    this->output()->copy(this->calc_array(), this->cuda_stream());
    cudaStreamSynchronize(this->cuda_stream());

    // shift ring buffer
    this->ring_buffer_pos() = (this->ring_buffer_pos() + 1) % this->buffer().size();
}
