"""Provision the isolated inspection service and its per-instance CA."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def proxy_service(workloads: dict, hosts: list[str]) -> dict:
    ca = "/etc/ai-assist-egress-ca.pem"
    bypass = ",".join(["localhost", "127.0.0.1", *workloads])
    peers = {
        port: name
        for port, name in [("8080", "egress-proxy"), ("8000", "dci-mcp-server"), ("4000", "litellm")]
        if name == "egress-proxy" or name in workloads
    }
    peers["443"] = "egress-proxy"
    dns_names = sorted({*workloads, *hosts, "egress-proxy"})
    for name, workload in list(workloads.items()):
        owner = f"{name}-network"
        workload.pop("networks")
        workload["network_mode"] = f"service:{owner}"
        workload["cap_drop"] = ["ALL"]
        workload["security_opt"] = ["no-new-privileges:true"]
        workload.setdefault("depends_on", {})[owner] = {"condition": "service_healthy"}
        workload["depends_on"]["egress-proxy"] = {"condition": "service_healthy"}
        workloads[owner] = {
            "image": "ai-assist-sandbox:latest",
            "user": "0:0",
            "entrypoint": ["python3.14", "-B", "/guard/network_guard.py"],
            "environment": {"WORKLOAD": name, "PEERS": json.dumps(peers), "DNS_NAMES": json.dumps(dns_names)},
            "volumes": ["./egress/guard:/guard:ro,z", "./audit:/audit:rw,z"],
            "networks": {"mcp-net": {"aliases": [name]}},
            "cap_drop": ["ALL"],
            "cap_add": ["NET_ADMIN"],
            "security_opt": ["no-new-privileges:true"],
            "healthcheck": {
                "test": ["CMD", "test", "-f", "/run/guard-ready"],
                "interval": "1s",
                "timeout": "2s",
                "retries": 15,
            },
        }
        workload.setdefault("volumes", []).append(f"./egress/public.pem:{ca}:ro,z")
        workload["environment"].update(
            {
                "HTTPS_PROXY": "http://egress-proxy:8080",
                "HTTP_PROXY": "http://egress-proxy:8080",
                "NO_PROXY": bypass,
                "SSL_CERT_FILE": ca,
                "GIT_SSL_CAINFO": ca,
                "REQUESTS_CA_BUNDLE": ca,
                "NODE_EXTRA_CA_CERTS": ca,
                "CURL_CA_BUNDLE": ca,
            }
        )
    return {
        "image": "docker.io/mitmproxy/mitmproxy:12.2.1",
        "entrypoint": ["python", "-B", "/policy/proxy.py"],
        "user": "0:0",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "volumes": ["./egress/policy:/policy:ro,Z", "./egress/ca:/ca:rw,Z", "./audit:/audit:rw,z"],
        "networks": {"upstream": {}, "mcp-net": {"aliases": hosts}},
        "sysctls": {"net.ipv4.ip_unprivileged_port_start": "443"},
        "healthcheck": {
            "interval": "2s",
            "timeout": "2s",
            "retries": 15,
            "test": ["CMD", "python", "-c", "import socket; socket.create_connection(('127.0.0.1',8080),1).close()"],
        },
    }


def write_proxy(instance: Path, policy: dict, rules: dict) -> None:
    base = instance / "egress"
    base.mkdir(parents=True, mode=0o700)
    private = base / "ca"
    private.mkdir(parents=True, mode=0o700)
    config = base / "policy"
    config.mkdir()
    (instance / "audit").mkdir(mode=0o700)
    guard = base / "guard"
    guard.mkdir()
    shutil.copyfile(Path(__file__).with_name("audit_writer.py"), guard / "audit_writer.py")
    shutil.copyfile(Path(__file__).parent / "sandbox_templates/network_guard.py.in", guard / "network_guard.py")
    with tempfile.TemporaryDirectory(dir=private) as staging:
        key = Path(staging) / "key.pem"
        cert = base / "public.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:3072",
                "-nodes",
                "-days",
                "365",
                "-subj",
                "/CN=ai-assist sandbox egress CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            capture_output=True,
            check=True,
        )
        (private / "mitmproxy-ca.pem").write_bytes(key.read_bytes() + cert.read_bytes())
        (private / "mitmproxy-ca.pem").chmod(0o600)
        cert.chmod(0o644)
    for module in ("audit_writer.py", "egress_guard.py"):
        shutil.copyfile(Path(__file__).with_name(module), config / module)
    shutil.copyfile(Path(__file__).parent / "sandbox_templates/egress_proxy.py.in", config / "proxy.py")
    (config / "policy.json").write_text(json.dumps({**rules, "destinations": policy}, indent=2))
