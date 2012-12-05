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

QMAKE_CXXFLAGS += -std=c++11 -I/usr/local/cuda/include
QMAKE_LDFLAGS += -lc++ -lfasteit -L/usr/local/cuda/lib64 -lcudart -lcublas
