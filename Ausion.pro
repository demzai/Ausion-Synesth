TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += C:\OpenCV3-2\opencv-build\install\include
LIBS += -LC:\OpenCV3-2\opencv-build\install\x86\mingw\lib \
                -lopencv_calib3d320.dll \
                -lopencv_core320.dll \
                -lopencv_features2d320.dll \
                -lopencv_flann320.dll \
                -lopencv_highgui320.dll \
                -lopencv_imgcodecs320.dll \
                -lopencv_imgproc320.dll \
                -lopencv_ml320.dll \
                -lopencv_objdetect320.dll \
                -lopencv_photo320.dll \
                -lopencv_shape320.dll \
                -lopencv_stitching320.dll \
                -lopencv_superres320.dll \
                -lopencv_video320.dll \
                -lopencv_videoio320.dll \
                -lopencv_videostab320.dll
