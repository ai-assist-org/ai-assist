FROM registry.fedoraproject.org/fedora:44@sha256:1ff9eb02e4e2d7fb1175becac7139b2960c4fdcccedce81c86ec25b98ef3124d

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
