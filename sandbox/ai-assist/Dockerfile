FROM registry.fedoraproject.org/fedora:44@sha256:86d20cf3b872310bafeb1e09152ffa95d0d76d79f08ec6be5f1765366bfd0adf

RUN dnf install -y \
        python3.14 python3.14-pip \
        jq git openssh-clients gnupg2 gh gitleaks && \
    dnf clean all

COPY . /opt/ai-assist
RUN python3.14 -m pip install --no-cache-dir /opt/ai-assist && \
    rm -rf /opt/ai-assist

RUN mkdir -p /workspace /host-config && chmod 777 /workspace /host-config

WORKDIR /workspace
ENTRYPOINT ["ai-assist"]
