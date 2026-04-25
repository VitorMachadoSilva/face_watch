#!/usr/bin/env bash
# FaceWatch — Script de instalação
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   FaceWatch — Instalação             ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python 3.9+
python3 --version >/dev/null 2>&1 || { echo "❌ Python 3 não encontrado. Instale Python 3.9+"; exit 1; }
PYVER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYVER" -lt 9 ]; then
    echo "❌ Python 3.9+ necessário (encontrado 3.$PYVER)"
    exit 1
fi

# Check pip
python3 -m pip --version >/dev/null 2>&1 || { echo "❌ pip não encontrado"; exit 1; }

# Check openssl
openssl version >/dev/null 2>&1 || { echo "❌ openssl não encontrado. Instale via: sudo apt install openssl"; exit 1; }

# Create virtualenv
if [ ! -d "venv" ]; then
    echo "→ Criando ambiente virtual..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "→ Instalando dependências..."
pip install -q --upgrade pip
pip install -q flask flask-sqlalchemy flask-bcrypt opencv-python-headless Pillow scikit-learn numpy

echo "→ Criando diretórios..."
mkdir -p database instance static/uploads/{faces,extras}

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "Para iniciar o servidor:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
