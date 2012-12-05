#-------------------------------------------------
#
# Project created by QtCreator 2012-12-05T12:02:47
#
#-------------------------------------------------

QT       += core gui opengl

TARGET = eitViewer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    image.cpp

HEADERS  += mainwindow.h \
    image.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -std=c++11

unix:!symbian: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lfasteit

INCLUDEPATH += $$PWD/../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../usr/local/include

unix:!symbian: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcudart

INCLUDEPATH += $$PWD/../../../../../../usr/local/cuda/include
DEPENDPATH += $$PWD/../../../../../../usr/local/cuda/include

unix:!symbian: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcublas

INCLUDEPATH += $$PWD/../../../../../../usr/local/cuda/include
DEPENDPATH += $$PWD/../../../../../../usr/local/cuda/include

OTHER_FILES += \
    nodes.txt \
    elements.txt \
    boundary.txt