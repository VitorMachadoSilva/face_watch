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

    # Tentativa 1: openssl CLI — subj apenas ASCII (evita bug de encoding no Windows)
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

    # Tentativa 2: biblioteca 'cryptography' (pura Python, ja inclusa com muitas libs)
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

    # Tentativa 3: adhoc Flask (funciona mas nao persiste entre reinicializacoes)
    print("[SSL] AVISO: usando SSL adhoc. Para certificado fixo, instale: pip install cryptography")
    return "adhoc", "adhoc"


def print_banner(local_ip, port):
    sep = "=" * 52
    print("\n" + sep)
    try:
        lines = [
            "  FaceWatch - Sistema de Identificacao",
            "  Facial para Agentes de Seguranca",
        ]
        # Tenta imprimir ASCII art; se falhar no terminal Windows, usa texto simples
        art = [
            "  FACEWATCH",
            "  Sistema de Identificacao Operacional",
        ]
        print("\n".join(art))
    except UnicodeEncodeError:
        print("  FACEWATCH")
    print(sep)
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

    ssl_result = ensure_ssl_cert()
    app = create_app()

    local_ip = get_local_ip()
    port     = 5443

    print_banner(local_ip, port)

    if ssl_result == ("adhoc", "adhoc"):
        ssl_ctx = "adhoc"
    else:
        cert_path, key_path = ssl_result
        ssl_ctx = (cert_path, key_path)

    app.run(
        host="0.0.0.0",
        port=port,
        ssl_context=ssl_ctx,
        debug=False,
        use_reloader=False,
        threaded=True,
    )