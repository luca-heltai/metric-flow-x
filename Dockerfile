# syntax=docker/dockerfile:1
ARG DEAL_II_IMAGE=dealii/dealii:v9.7.1-noble@sha256:28a40c95e76879c717a96ea3f5c87cc00ec1d7a4ce9d250ab936fb723eab6995
FROM ${DEAL_II_IMAGE} AS build
USER root
WORKDIR /src
COPY . .
ARG BUILD_JOBS=2
RUN cmake -S . -B /build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/metric-flow-x \
    && cmake --build /build --target blood_flow -j "${BUILD_JOBS}" \
    && cmake --install /build

# Keep the matching deal.II SDK and shared dependencies for library consumers.
FROM ${DEAL_II_IMAGE}
USER root
COPY --from=build /opt/metric-flow-x /opt/metric-flow-x
COPY LICENSE.md /opt/metric-flow-x/share/LICENSE.md
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="MetricFlowX" \
      org.opencontainers.image.source="https://github.com/luca-heltai/metric-flow-x" \
      org.opencontainers.image.revision="${VCS_REF}"
ENV PATH="/opt/metric-flow-x/bin:${PATH}" \
    CMAKE_PREFIX_PATH="/opt/metric-flow-x" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN mkdir /work && chmod 1777 /work
WORKDIR /work
USER 1000:1000
ENTRYPOINT ["/opt/metric-flow-x/bin/blood_flow"]
