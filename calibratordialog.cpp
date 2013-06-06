#include "calibratordialog.h"
#include "ui_calibratordialog.h"

CalibratorDialog::CalibratorDialog(Calibrator* calibrator, QWidget *parent)
    : QDialog(parent), ui(new Ui::CalibratorDialog) {
    // init ui
    ui->setupUi(this);
    this->setFixedSize(this->geometry().width(), this->geometry().height());

    // fill text boxes with calibrator data
    this->ui->regularization_factor_box->setText(QString("%1").arg(
        calibrator->fasteit_solver()->inverse_solver()->regularization_factor()));
    this->ui->step_size_box->setText(QString("%1").arg(calibrator->step_size()));
    this->ui->filter_step_size_box->setText(QString("%1").arg(
        calibrator->filter()->step_size()));

    // update calibrator settings on acceptance
    connect(this, &QDialog::accepted, [=]() {
        calibrator->fasteit_solver()->inverse_solver()->regularization_factor() =
            this->ui->regularization_factor_box->text().toDouble();
        calibrator->restart(this->ui->step_size_box->text().toInt());
        calibrator->filter()->restart(this->ui->filter_step_size_box->text().toInt());
    });
}

CalibratorDialog::~CalibratorDialog() {
    delete ui;
}
