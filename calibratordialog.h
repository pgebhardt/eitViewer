#ifndef CALIBRATORDIALOG_H
#define CALIBRATORDIALOG_H

#include <QDialog>
#include "calibrator.h"

namespace Ui {
class CalibratorDialog;
}

class CalibratorDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit CalibratorDialog(Calibrator* calibrator, QWidget* parent=nullptr);
    ~CalibratorDialog();
    
private:
    Ui::CalibratorDialog *ui;
};

#endif // CALIBRATORDIALOG_H
