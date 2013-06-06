#ifndef CALIBRATORDIALOG_H
#define CALIBRATORDIALOG_H

#include <QDialog>

namespace Ui {
class CalibratorDialog;
}

class CalibratorDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit CalibratorDialog(QWidget *parent = 0);
    ~CalibratorDialog();
    
private:
    Ui::CalibratorDialog *ui;
};

#endif // CALIBRATORDIALOG_H
