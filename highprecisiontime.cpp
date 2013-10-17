#include "highprecisiontime.h"

HighPrecisionTime::HighPrecisionTime(QObject *parent) :
    QObject(parent) {
    this->restart();
}

void HighPrecisionTime::restart() {
    // reset start time
    this->start_time_ = std::chrono::high_resolution_clock::now();
}

double HighPrecisionTime::elapsed() {
    // calculate time difference to start time
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - this->start_time_)
        .count();
}
