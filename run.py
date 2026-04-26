#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FaceWatch — Entry point (ajustado para Render)
"""

import os
from app import create_app

# Cria o app Flask
app = create_app()

if __name__ == "__main__":
    # Render define a porta via variável de ambiente
    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 Iniciando servidor na porta {port}...")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )