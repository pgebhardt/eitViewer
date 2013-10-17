#ifndef HIGHPRECISIONTIME_H
#define HIGHPRECISIONTIME_H

#include <QObject>
#include <chrono>

class HighPrecisionTime : public QObject
{
    Q_OBJECT
public:
    explicit HighPrecisionTime(QObject *parent = 0);
    void restart();
    double elapsed();

private:
   std::chrono::high_resolution_clock::time_point start_time_;
};

#endif // HIGHPRECISIONTIME_H
