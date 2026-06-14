FROM ai-assist-sandbox:latest

USER root

RUN dnf install -y \
        golang ansible-core \
        python3-pip python3-devel uv \
        shellcheck yamllint \
        gcc make cmake && \
    dnf clean all
