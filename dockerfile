FROM  gocv/opencv:4.11.0-ubuntu-20.04

WORKDIR /imagehandler

RUN apt-get update && apt-get install -y g++ libpqxx-dev libopencv-dev pkg-config
    
COPY . .
    
RUN OPENCV_CFLAGS=$(pkg-config --cflags opencv4) && \
    OPENCV_LIBS=$(pkg-config --libs opencv4) && \
    g++ -std=c++17 -o crowhandler ImageHandlerCrow.cpp -Iinclude -lpqxx $OPENCV_CFLAGS $OPENCV_LIBS -pthread

CMD ["./crowhandler"]