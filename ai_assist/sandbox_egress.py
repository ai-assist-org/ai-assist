"""Provision the isolated inspection service and its per-instance CA."""

import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def configure_egress_subnet(compose: dict, subnet: str, instance_name: str) -> None:
    """Set up internal service discovery on Podman bridges without DNS."""
    services = compose["services"]
    if "egress-proxy" not in services:
        raise ValueError("--egress-subnet requires egress-proxy")
    network = ipaddress.IPv4Network(subnet)
    owners = {name.removesuffix("-network"): svc for name, svc in services.items() if name.endswith("-network")}
    owners["egress-proxy"] = services["egress-proxy"]
    private = any(
        network.subnet_of(ipaddress.IPv4Network(block)) for block in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    )
    if not private or network.num_addresses < len(owners) + 3:
        raise ValueError("Choose an unused private IPv4 subnet with space for the gateway and all sandbox services")
    compose["networks"]["mcp-net"]["ipam"] = {"config": [{"subnet": str(network)}]}
    peers = {}
    for index, (name, owner) in enumerate(owners.items(), start=2):
        peers[name] = str(network[index])
        owner["networks"]["mcp-net"]["ipv4_address"] = peers[name]
    proxy = owners["egress-proxy"]
    aliases = proxy["networks"]["mcp-net"]["aliases"]
    local_entries = [f"{name}:{ip}" for name, ip in peers.items()]
    proxy["extra_hosts"] = local_entries
    for name, owner in owners.items():
        if name != "egress-proxy":
            owner["extra_hosts"] = local_entries + [f"{host}:{peers['egress-proxy']}" for host in aliases]
            owner["environment"]["DNS_NAMES"] = "[]"
            workload = services[name]
            if "userns_mode" in workload:
                # runc needs the owner of the network namespace for sysfs
                # and exec (including health checks). Preserve host UID mapping.
                owner["container_name"] = f"{instance_name}_{name}-network_1"
                owner["userns_mode"] = workload["userns_mode"]
                user = "1001:1001" if name == "dci-mcp-server" else f"{os.getuid()}:{os.getgid()}"
                owner["user"] = workload["user"] = user
                owner["tmpfs"] = ["/run:rw,nosuid,nodev,mode=1777"]
                workload["userns_mode"] = f"container:{owner['container_name']}"


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
        # Older OpenSSL versions append -addext to site defaults, duplicating
        # basicConstraints and producing a CA that strict clients reject.
        config_file = Path(staging) / "openssl.cnf"
        config_file.write_text(
            "[req]\ndistinguished_name=dn\nx509_extensions=ca\n[dn]\n[ca]\n"
            "basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n"
        )
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
                "-config",
                str(config_file),
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
