FROM registry.fedoraproject.org/fedora:44@sha256:3988086adef2975a8a3a6c15634e21625efebc6ac043c83f02f619efff6c5d11

RUN dnf install -y \
        python3.14 python3.14-pip \
        jq git openssh-clients gnupg2 gh && \
    dnf clean all

COPY . /opt/ai-assist
RUN python3.14 -m pip install --no-cache-dir /opt/ai-assist && \
    rm -rf /opt/ai-assist

RUN mkdir -p /workspace /host-config && chmod 777 /workspace /host-config

WORKDIR /workspace
ENTRYPOINT ["ai-assist"]
