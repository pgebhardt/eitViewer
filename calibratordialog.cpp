#include "calibratordialog.h"
#include "ui_calibratordialog.h"

CalibratorDialog::CalibratorDialog(Calibrator* calibrator, QWidget *parent)
    : QDialog(parent), ui(new Ui::CalibratorDialog) {
    // init ui
    ui->setupUi(this);
    this->setFixedSize(this->geometry().width(), this->geometry().height());

    // fill text boxes with calibrator data
    this->ui->regularization_factor_box->setText(QString("%1").arg(
        calibrator->eit_solver()->inverse_solver()->regularization_factor()));
    this->ui->calibrator_interval_box->setText(QString::number((double)calibrator->step_size() * 1e-3, 'f', 1));
    this->ui->filter_constant_box->setText(QString::number(calibrator->filterConstant(), 'f', 1));

    // update calibrator settings on acceptance
    connect(this, &QDialog::accepted, [=]() {
        calibrator->eit_solver()->inverse_solver()->regularization_factor() =
            this->ui->regularization_factor_box->text().toDouble();
        calibrator->step_size() = (int)(this->ui->calibrator_interval_box->text().toDouble() * 1e3);
        calibrator->filterConstant() = this->ui->filter_constant_box->text().toDouble();
    });
}

CalibratorDialog::~CalibratorDialog() {
    delete ui;
}
