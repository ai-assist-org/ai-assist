FROM fedora:44

RUN dnf install -y \
        python3.14 python3.14-pip \
        jq && \
    dnf clean all

COPY . /opt/dci-mcp-server
RUN python3.14 -m pip install --no-cache-dir /opt/dci-mcp-server && \
    rm -rf /opt/dci-mcp-server

EXPOSE 8001

ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8001

CMD ["python3.14", "-m", "mcp_server.main"]
