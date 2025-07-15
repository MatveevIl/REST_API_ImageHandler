FROM ubuntu:20.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/opencv_build

ARG OPENCV_VERSION=4.11.0
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv

WORKDIR /tmp/opencv_build/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D BUILD_EXAMPLES=OFF .. && \
    make -j2 && \
    make install && \
    ldconfig

WORKDIR /imagehandler

COPY . .

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

RUN OPENCV_CFLAGS=$(pkg-config --cflags opencv4) && \
    OPENCV_LIBS=$(pkg-config --libs opencv4) && \
    g++ -std=c++17 -o crowhandler ImageHandlerCrow.cpp -Iinclude -lpqxx $OPENCV_CFLAGS $OPENCV_LIBS

FROM ubuntu:20.04 AS final

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
     apt-get update && \
    apt-get install -y g++ libpqxx-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /imagehandler/crowhandler /app/crowhandler

COPY --from=builder /usr/local/lib/* /usr/lib/

ENV LD_LIBRARY_PATH=/usr/lib

WORKDIR /app

CMD ["./crowhandler"]
