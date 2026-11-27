FROM registry.fedoraproject.org/fedora:44@sha256:0daf6a73fb6ad75dcd62d9a5270e58572f1aea95bb9331db7c2eedd6e47c9cb5

RUN dnf install -y \
        python3.14 python3.14-pip \
        jq git openssh-clients gnupg2 gh gitleaks iproute nftables libnetfilter_queue && \
    dnf clean all

COPY . /opt/ai-assist
RUN python3.14 -m pip install --no-cache-dir /opt/ai-assist && \
    rm -rf /opt/ai-assist

RUN mkdir -p /workspace /host-config && chmod 777 /workspace /host-config

WORKDIR /workspace
ENTRYPOINT ["ai-assist"]
