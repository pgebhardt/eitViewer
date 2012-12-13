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
    measurementsystem.cpp

HEADERS  += mainwindow.h \
    image.h \
    measurementsystem.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -std=c++11

macx {
    QMAKE_CXXFLAGS += -stdlib=libc++
    QMAKE_LIBS += -lc++
    QMAKE_LIBS += -Xlinker -rpath /usr/local/cuda/lib
    QMAKE_CXX = clang++
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.7
}

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

RESOURCES += \
    mesh.qrc
