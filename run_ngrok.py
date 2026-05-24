#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FaceWatch — Entry point"""

import os
import sys
import socket
import subprocess
from pathlib import Path

# Garantir que o diretorio do script seja o CWD
os.chdir(Path(__file__).parent)

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

from app import create_app


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_ssl_cert():
    os.makedirs("instance", exist_ok=True)
    cert_path = os.path.join("instance", "cert.pem")
    key_path  = os.path.join("instance", "key.pem")

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    print("[SSL] Gerando certificado auto-assinado...")

    subj = "/CN=FaceWatch/O=Security/C=BR"
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509",
                "-newkey", "rsa:2048",
                "-keyout", key_path,
                "-out",    cert_path,
                "-days",   "3650",
                "-nodes",
                "-subj",   subj,
            ],
            check=True,
            capture_output=True,
        )
        print("[SSL] Certificado gerado via openssl CLI.")
        return cert_path, key_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime

        print("[SSL] Usando lib 'cryptography' para gerar certificado...")

        priv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME,       "FaceWatch"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Security"),
            x509.NameAttribute(NameOID.COUNTRY_NAME,      "BR"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(priv_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False,
            )
            .sign(priv_key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(priv_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print("[SSL] Certificado gerado com sucesso.")
        return cert_path, key_path

    except ImportError:
        pass

    print("[SSL] AVISO: usando SSL adhoc. Para certificado fixo, instale: pip install cryptography")
    return "adhoc", "adhoc"


def print_banner(local_ip, port, ngrok_mode=False):
    sep = "=" * 52
    print("\n" + sep)
    print("  FACEWATCH")
    print("  Sistema de Identificacao Operacional")
    print(sep)
    if ngrok_mode:
        print(f"\n  Modo NGROK ativo (HTTP sem SSL)")
        print(f"  Local   : http://localhost:{port}")
        print(f"\n  Rode em outro terminal:")
        print(f"  .\\ngrok.exe http {port}")
        print(f"\n  O link HTTPS do Ngrok e o que voce compartilha")
    else:
        print(f"\n  Desktop : https://localhost:{port}")
        print(f"  Mobile  : https://{local_ip}:{port}")
        print(f"\n  IMPORTANTE: No browser, clique em 'Avancado' > 'Prosseguir'")
        print(f"              para aceitar o certificado auto-assinado.")
    print(sep + "\n")


if __name__ == "__main__":
    # UTF-8 no terminal Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ── Modo Ngrok: sem SSL, porta 5000 ─────────────────────────────────────
    # Para ativar: defina a variável de ambiente NO_SSL=1
    #   PowerShell: $env:NO_SSL="1"; python run.py
    #   CMD:        set NO_SSL=1 && python run.py
    ngrok_mode = bool(os.environ.get("NO_SSL"))

    app      = create_app()
    local_ip = get_local_ip()

    if ngrok_mode:
        port    = 5000
        ssl_ctx = None
    else:
        port       = 5443
        ssl_result = ensure_ssl_cert()
        ssl_ctx    = "adhoc" if ssl_result == ("adhoc", "adhoc") else (ssl_result[0], ssl_result[1])

    print_banner(local_ip, port, ngrok_mode=ngrok_mode)

    app.run(
        host="0.0.0.0",
        port=port,
        ssl_context=ssl_ctx,
        debug=False,
        use_reloader=False,
        threaded=True,
    )