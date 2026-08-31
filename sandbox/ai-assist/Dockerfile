FROM fedora:44@sha256:43b29f65a41eb9c35e1cd5323e3bdf3b655c2357a9f4f1ff2f9c2798e5045d80

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
