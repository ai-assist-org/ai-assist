"""Exercise the generated inspector against an isolated HTTPS and smart-Git origin."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ai_assist.sandbox import DEFAULT_IMAGE, _compose_cmd, sandbox_init

pytestmark = pytest.mark.integration


def test_inspected_egress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not shutil.which("podman") or not shutil.which("podman-compose"):
        pytest.skip("requires rootless podman and podman-compose")
    for image in (DEFAULT_IMAGE, "docker.io/mitmproxy/mitmproxy:12.2.1"):
        if subprocess.run(["podman", "image", "exists", image], check=False).returncode:
            pytest.skip(f"requires local image {image}")
    monkeypatch.setenv("AI_ASSIST_INSTANCES_DIR", str(tmp_path))
    name = f"egress-live-{os.getpid()}"
    sandbox_init(name, {"egress-proxy", "litellm"})
    instance = tmp_path / name
    (instance / ".env").write_text("LITELLM_MASTER_KEY=sk-fixture\n")
    (instance / "sandbox/.ai-assist/litellm_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": "fixture",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": "test",
                            "api_base": "https://model.test/v1",
                        },
                    }
                ],
                "router_settings": {"num_retries": 0},
                "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            }
        )
    )
    fixture = instance / "fixture"
    fixture.mkdir()
    shutil.copyfile(Path(__file__).parent / "fixtures/egress_origin.py", fixture / "origin.py")
    shutil.copyfile(Path(__file__).parent / "fixtures/egress_ech.go", fixture / "ech.go")
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=full.test",
            "-keyout",
            "server.key",
            "-out",
            "server.csr",
        ],
        cwd=fixture,
        check=True,
        capture_output=True,
    )
    (fixture / "extensions").write_text("subjectAltName=DNS:full.test,DNS:model.test,DNS:tracker.test\n")
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            "server.csr",
            "-CA",
            "../egress/public.pem",
            "-CAkey",
            "../egress/ca/mitmproxy-ca.pem",
            "-set_serial",
            "10",
            "-days",
            "1",
            "-extfile",
            "extensions",
            "-out",
            "server.pem",
        ],
        cwd=fixture,
        check=True,
        capture_output=True,
    )
    policy = {
        h: {"profile": p, "addresses": ["origin"]}
        for h, p in [("full.test", "full_dlp"), ("model.test", "secrets_only"), ("tracker.test", "none")]
    }
    policy_file = instance / "egress/policy/policy.json"
    policy_file.write_text(json.dumps({**json.loads(policy_file.read_text()), "destinations": policy}))
    manifest = instance / "compose.yaml"
    stack = yaml.safe_load(manifest.read_text())
    stack["networks"]["upstream"]["internal"] = True
    proxy = stack["services"]["egress-proxy"]
    proxy["networks"]["mcp-net"]["aliases"] = list(policy)
    for owner in ("ai-assist-network", "litellm-network"):
        guard_env = stack["services"][owner]["environment"]
        guard_env["DNS_NAMES"] = json.dumps(json.loads(guard_env["DNS_NAMES"]) + list(policy))
    proxy["environment"] = {"SSL_CERT_FILE": "/fixture-ca.pem"}
    proxy["volumes"].append("./egress/public.pem:/fixture-ca.pem:ro,z")
    stack["services"]["origin"] = {
        "image": DEFAULT_IMAGE,
        "networks": ["upstream"],
        "volumes": ["./fixture:/fixture:ro,z"],
        "entrypoint": ["python3.14", "/fixture/origin.py"],
    }
    stack["services"]["ech-client"] = {
        "image": "ai-assist-dev:latest",
        "network_mode": "service:ai-assist-network",
        "cap_drop": ["ALL"],
        "volumes": ["./fixture/ech.go:/ech.go:ro,z"],
        "environment": {"GOCACHE": "/tmp/gocache", "GOTOOLCHAIN": "local"},
        "entrypoint": ["go", "run", "/ech.go"],
    }
    manifest.write_text(yaml.safe_dump(stack))
    compose = _compose_cmd(instance)

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run([*compose, *args], cwd=instance, capture_output=True, text=True, timeout=90, check=check)

    def client(script: str) -> None:
        result = run(
            "run", "--rm", "-T", "--no-deps", "--entrypoint", "python3.14", "ai-assist", "-c", script, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    try:
        run("up", "-d", "origin", "egress-proxy", "ai-assist-network", "litellm-network")
        upstream_ip = run(
            "exec", "-T", "origin", "python3.14", "-c", "import socket;print(socket.gethostbyname('origin'))"
        ).stdout.strip()
        client(f"""
import errno, socket
for socktype, proto, port in [(socket.SOCK_STREAM, 0, 443), (socket.SOCK_STREAM, 0, 22),
                              (socket.SOCK_DGRAM, 0, 443), (socket.SOCK_DGRAM, socket.IPPROTO_ICMP, 0)]:
    try:
        with socket.socket(socket.AF_INET, socktype, proto) as s:
            s.settimeout(2)
            s.connect(({upstream_ip!r},port))
            if socktype == socket.SOCK_DGRAM:
                s.send(bytes([8,0,0,0,0,1,0,1]) if proto else b'test')
                s.recv(64)
    except OSError as e:
        assert e.errno in (errno.ENETUNREACH,errno.EHOSTUNREACH,errno.EACCES,errno.EPERM), e
    else: raise AssertionError((socktype,proto,port))
""")
        client("""
import httpx, time
from pathlib import Path
for _ in range(20):
    try:
        if httpx.get('https://model.test/',timeout=2).status_code == 200: break
    except httpx.HTTPError: pass
    time.sleep(.25)
else: raise RuntimeError('inspection stack did not become ready')
assert not Path('/ca/mitmproxy-ca.pem').exists() and not Path('/policy/policy.json').exists()
import ssl
with httpx.Client(trust_env=False,verify=ssl.create_default_context(cafile='/etc/ai-assist-egress-ca.pem')) as direct:
    assert direct.get('https://model.test/').status_code == 200
for host in ['example.com','192.0.2.1']:
    try: httpx.get('https://'+host+'/',timeout=2)
    except httpx.ProxyError as exc: assert '403' in str(exc)
    else: raise AssertionError(host)
""")

        def accepted() -> str:
            return run(
                "exec",
                "-T",
                "origin",
                "python3.14",
                "-c",
                "from pathlib import Path;print(Path('/tmp/accepted').stat().st_size)",
            ).stdout

        before = accepted()
        client("""
import socket,ssl
for sni in (None, 'other.test'):
    with socket.create_connection(('egress-proxy',8080),2) as sock:
        sock.sendall(b'CONNECT full.test:443 HTTP/1.1\\r\\nHost: full.test:443\\r\\n\\r\\n')
        assert b'200' in sock.recv(4096)
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname=False
        context.load_verify_locations('/etc/ai-assist-egress-ca.pem')
        try: context.wrap_socket(sock,server_hostname=sni)
        except (ssl.SSLError,ConnectionResetError): pass
        else: raise AssertionError('unexpected TLS connection')
""")
        assert accepted() == before
        audit_path = instance / "audit/egress.jsonl"
        offset = audit_path.stat().st_size
        result = run("run", "--rm", "-T", "--no-deps", "ech-client", check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        events = [json.loads(line) for line in audit_path.read_bytes()[offset:].splitlines()]
        assert [r["rule_id"] for r in events if r["destination"] == "full.test"] == ["destination"]
        assert accepted() == before
        client("""
import httpx
secret = b'AKIA' + b'Q'*16
for host, body, code in [('full.test', b'CILAB-123',403), ('model.test', b'CILAB-123',200),
                        ('model.test',secret,403), ('tracker.test',secret,200), ('full.test',b'public',200)]:
    r=httpx.post('https://'+host+'/',content=body)
    assert r.status_code == code, r.status_code
    assert r.content == body if code == 200 else body not in r.content
r=httpx.post('https://full.test/',headers={'Content-Encoding':'gzip'},content=b'broken')
assert r.status_code == 403 and 'not-inspected' in r.text
with httpx.stream('GET', 'https://model.test/events',timeout=4) as response:
    lines=response.iter_lines()
    assert next(lines)=='data: first'
    httpx.get('https://model.test/release')
    assert 'data: last' in list(lines)
""")
        client("""
import subprocess,tempfile,os
with tempfile.TemporaryDirectory() as root:
    os.chdir(root)
    def git(*args,check=True):
        return subprocess.run(['git',*args],check=check,capture_output=True,text=True)
    git('init');git('config','user.name','fixture');git('config','user.email','fixture@example.test')
    git('commit','--allow-empty','-m','public');git('push','https://full.test/repo.git','HEAD:refs/heads/main')
    open('file','w').write('CILAB-123');git('add','file');git('commit','-m','second')
    r=git('push','https://full.test/repo.git','HEAD:refs/heads/main',check=False)
    assert r.returncode and 'internal-ticket' in r.stderr and 'CILAB-123' not in r.stderr,r.stderr
    open('file','w').write('public replacement');git('add','file');git('commit','--amend','--no-edit')
    git('push','https://full.test/repo.git','HEAD:refs/heads/main')
""")
        audit = (instance / "audit/egress.jsonl").read_text()
        assert "CILAB-123" not in audit and "AKIA" not in audit
        records = [json.loads(line) for line in audit.splitlines()]
        assert {"internal-ticket", "aws-key", "not-inspected"} <= {r["rule_id"] for r in records}
        assert sum(r["rule_id"] == "internal-ticket" for r in records) == 2
        assert sum(r["rule_id"] == "aws-key" for r in records) == 1
        direct = [(r["protocol"], r["port"]) for r in records if r["destination"] == upstream_ip]
        assert sorted(direct) == sorted([(6, 443), (6, 22), (17, 443), (1, None)])
        assert {r["protocol"] for r in records if r["rule_id"] == "direct-network"} >= {6, 17}
        offset = audit_path.stat().st_size
        client("""
import httpx
with httpx.Client() as session:
    for body, status in [("public", 200), ("CILAB-123", 403), ("CILAB-123", 403)]:
        assert session.post("https://full.test/",json={"body":body}).status_code == status
""")
        window = [json.loads(line) for line in audit_path.read_bytes()[offset:].splitlines()]
        assert all({"timestamp", "source", "destination", "decision", "rule_id", "profile"} <= r.keys() for r in window)
        requests = []
        for record in window:
            if record["rule_id"] == "direct-network":
                # The kernel independently solicits IPv6 routers on the dummy interface.
                assert (record["protocol"], record["destination"], record["decision"]) == (58, "ff02::2", "refused")
            else:
                requests.append((record["decision"], record["rule_id"]))
        assert requests == [("allowed", None), ("refused", "internal-ticket"), ("refused", "internal-ticket")]
        client("""
import json, os, subprocess, tempfile
from pathlib import Path
env = dict(os.environ, GH_ENTERPRISE_TOKEN="ghp_" + "X"*36, GH_PROMPT_DISABLED="1")
with tempfile.TemporaryDirectory() as root:
    path = Path(root) / "request"
    for option in ("--input", "-F"):
        for allowed, text in ((True, "public Markdown"), (False, "CILAB-123")):
            path.write_text(json.dumps({"text": text}) if option == "--input" else text)
            args = [option, str(path) if option == "--input" else "text=@" + str(path)]
            result = subprocess.run(["gh", "api", "--hostname", "full.test", "/render", *args],
                                    env=env, capture_output=True, text=True, timeout=15)
            assert (result.returncode == 0) == allowed, (result.stdout, result.stderr)
            if allowed:
                assert json.loads(result.stdout)["text"] == text
            else:
                assert "internal-ticket" in result.stdout + result.stderr
import httpx
for target, headers in [("/api/v3/render/CILAB-123", {}), ("/api/v3/render?q=CILAB%2D123", {}),
                        ("/api/v3/render", {"X-Review": "CILAB-123"})]:
    reply = httpx.get("https://full.test" + target, headers=headers)
    assert reply.status_code == 403 and "internal-ticket" in reply.text
""")
        count = run("exec", "-T", "origin", "wc", "-c", "/tmp/render-count")
        assert int(count.stdout.split()[0]) == 2
        run("up", "-d", "--no-deps", "litellm")
        client("""
import httpx, time
url = "http://litellm:4000"
for _ in range(100):
    try:
        if httpx.get(url + "/health/liveliness", timeout=1).is_success: break
    except httpx.HTTPError: pass
    time.sleep(.5)
else: raise AssertionError("LiteLLM did not start")
for prompt, allowed in [("public", True), ("AKIA" + "Z"*16, False)]:
    response = httpx.post(url + "/v1/chat/completions", headers={"Authorization": "Bearer sk-fixture"},
        json={"model":"fixture", "messages":[{"role":"user","content":prompt}]}, timeout=15)
    if allowed:
        assert response.is_success, response.text
        assert response.json()["choices"][0]["message"]["content"] == "fixture reply"
    else: assert not response.is_success
""")
        decisions = [json.loads(line) for line in (instance / "audit/egress.jsonl").read_text().splitlines()]
        assert {r["rule_id"] for r in decisions if r["source"] == "litellm" and r["destination"] == "model.test"} >= {
            None,
            "aws-key",
        }
        run("stop", "egress-proxy")
        client("""
import httpx
try: httpx.get('https://model.test/',timeout=2)
except httpx.HTTPError: pass
else: raise AssertionError('origin reachable after proxy stopped')
""")
    finally:
        run("down", "--remove-orphans")
