#-------------------------------------------------
#
# Project created by QtCreator 2012-12-05T12:02:47
#
#-------------------------------------------------

QT       += core gui opengl network

TARGET = eitViewer
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    image.cpp \
    measurementsystem.cpp \
    jsonobject.cpp

HEADERS  += mainwindow.h \
    image.h \
    measurementsystem.h \
    jsonobject.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -std=c++11

macx {
    QMAKE_CXXFLAGS += -stdlib=libc++
    QMAKE_LIBS += -lc++
    QMAKE_LIBS += -Xlinker -rpath /usr/local/cuda/lib
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.7
}

unix:!symbian: LIBS += -L/usr/local/cuda/lib64/ -lcudart -lcublas

INCLUDEPATH += /usr/local/cuda/include
DEPENDPATH += /usr/local/cuda/include

OTHER_FILES +=

RESOURCES +=

unix:!symbian: LIBS += -Llibs/lib/ -lfasteit

INCLUDEPATH += libs/include
DEPENDPATH += libs/include
