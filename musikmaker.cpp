#include <cmath>
#include <iostream>
#include "musikmaker.h"

MusikMaker::MusikMaker(std::shared_ptr<fastEIT::Model<fastEIT::basis::Linear>> model,
                       QObject *parent) :
    QObject(parent), model_(model) {
    // load nodes
    this->sounds()["A"] = new QSound("wave/test.wav");
    this->sounds()["B"] = new QSound("wave/test.wav");
    this->sounds()["C"] = new QSound("wave/test.wav");
    this->sounds()["D"] = new QSound("wave/test.wav");
    this->sounds()["E"] = new QSound("wave/test.wav");
    this->sounds()["F"] = new QSound("wave/test.wav");
    this->sounds()["G"] = new QSound("wave/test.wav");
}

std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> MusikMaker::getPosition(
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> gamma,
    fastEIT::dtype::real threshold) {
    // get maximum element
    fastEIT::dtype::real value = 0.0;
    fastEIT::dtype::index max_index = 0;

    for (fastEIT::dtype::index element = 0; element < gamma->rows(); ++element) {
        // check value
        if (std::abs((*gamma)(element, 0)) > value) {
            value = std::abs((*gamma)(element, 0));
            max_index = element;
        }
    }

    // get element nodes
    auto nodes = this->model()->mesh()->elementNodes(max_index);

    // calc midpoint
    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> mid_point =
        std::make_tuple(0.0f, 0.0f);

    if (value >= threshold) {
        for (auto node : nodes) {
            // add node to midpoint
            std::get<0>(mid_point) += std::get<0>(node) / 3.0;
            std::get<1>(mid_point) += std::get<1>(node) / 3.0;
        }
    }

    return mid_point;
}

QString MusikMaker::getNode(
    std::tuple<fastEIT::dtype::real, fastEIT::dtype::real> position) {
    // convert to polar coordinates
    fastEIT::dtype::real radius, angle;
    std::tie(radius, angle) = fastEIT::math::polar(position);

    // check radius
    QString node = "";
    if (radius < this->model()->mesh()->radius() / 3.0) {
        node = "X";
    } else {
        node = "A" + (char)(angle * 3.5 / M_PI);
    }

    return node;
}

void MusikMaker::playNode(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> gamma,
              fastEIT::dtype::real threshold) {
    // get position
    auto position = this->getPosition(gamma, threshold);

    // get node
    auto node = this->getNode(position);

    // play node if new
    if ((node != "X") && (node != this->previous_node())) {
       this->sounds()[node]->play();
    }
}
