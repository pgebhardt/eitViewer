#include "calibratordialog.h"
#include "ui_calibratordialog.h"

CalibratorDialog::CalibratorDialog(QWidget *parent)
    : QDialog(parent), ui(new Ui::CalibratorDialog) {
    ui->setupUi(this);
}

CalibratorDialog::~CalibratorDialog() {
    delete ui;
}
