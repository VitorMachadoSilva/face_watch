# FaceWatch 🔍

Sistema web de identificação e cadastro de pessoas para uso operacional por agentes de segurança.

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Backend | Python 3, Flask, Flask-SQLAlchemy (SQLite) |
| Reconhecimento | OpenCV (Haar Cascade + HOG + CLAHE + KNN) |
| Frontend | HTML/CSS/JS puro (sem frameworks) |
| SSL | Certificado auto-assinado via OpenSSL |

## Estrutura do Projeto

```
facewatch/
├── run.py                    # Ponto de entrada
├── requirements.txt
├── install.sh
├── app/
│   ├── __init__.py           # App factory (create_app)
│   ├── models.py             # User, Person, Occurrence
│   ├── routes/
│   │   ├── auth.py           # /api/login, /api/logout, /api/me
│   │   ├── api.py            # Todos endpoints JSON
│   │   └── pages.py          # Rotas HTML (render_template)
│   └── services/
│       └── face.py           # Pipeline completo de reconhecimento
├── templates/
│   ├── base.html             # Layout base, CSS global, JS utilitários
│   ├── login.html
│   ├── identify.html         # Scanner em tempo real
│   ├── register.html         # Cadastro de pessoa
│   ├── search.html           # Busca paginada
│   └── person.html           # Ficha completa + ocorrências
├── static/
│   └── uploads/
│       ├── faces/            # Fotos de rosto
│       └── extras/           # Fotos extras (corpo, tatuagens)
├── database/
│   ├── facewatch.db          # SQLite (gerado automaticamente)
│   └── knn_model.pkl         # Modelo KNN serializado
└── instance/
    ├── cert.pem              # Certificado SSL (gerado automaticamente)
    └── key.pem
```

## Instalação e Uso

### Linux / macOS

```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
python run.py
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

### Acesso

O servidor inicia na porta **5443** (HTTPS).  
- Desktop: `https://localhost:5443`  
- Mobile (mesma rede): `https://192.168.X.X:5443`

> ⚠️ **Aceite o certificado auto-assinado** no browser na primeira vez.

## Usuários Padrão

| Usuário | Senha | Papel |
|---------|-------|-------|
| admin | admin123 | Administrador |
| policial | policial123 | Agente |

## Pipeline de Reconhecimento Facial

```
Imagem (câmera / upload / base64)
    ↓
PIL.ImageOps.exif_transpose()    ← Corrige rotação EXIF
    ↓
Haar Cascade (frontal + alt2 + profile)
  params progressivos: minNeighbors 4→1, minSize 40→10px
    ↓ (se falhar)
Crop central 10%–90%             ← Fallback — nunca rejeita
    ↓
cv2.resize(64×64)
    ↓
CLAHE equalização
    ↓
HOG descriptor (64×64, 9 bins)   ← Estável entre compressões JPEG
    ↓
Histograma de intensidade (32 bins)
    ↓
Concatenar + L2-normalizar
    ↓
KNeighborsClassifier(k=3, metric='euclidean', weights='distance')
    ↓
Threshold auto-calibrado         ← mean + max(std×3, mean×2)
```

**Regra de ouro:** `_features_from_array()` é chamado **identicamente** em treino e inferência.

## Endpoints da API

```
POST   /api/login
POST   /api/logout
GET    /api/me

GET    /api/stats
GET    /api/persons               ?q=&risk=&status=&page=&per_page=
POST   /api/persons
GET    /api/persons/<id>
PUT    /api/persons/<id>
DELETE /api/persons/<id>

POST   /api/persons/<id>/photo/face
POST   /api/persons/<id>/photo/extra
POST   /api/persons/<id>/occurrences

POST   /api/recognize             { image_b64: "..." }
POST   /api/recognize/realtime    { image_b64: "..." }
POST   /api/retrain

GET    /api/uploads/<path>        Serve arquivos de foto
```

## Funcionalidades

### Identificar
- Câmera ao vivo via `getUserMedia` (HTTPS obrigatório)
- Scan automático a cada 1.5s com HUD overlay
- Botão de captura frame completo
- Upload de arquivo para identificação offline
- Painel de resultado com confiança, risco, status e atalhos

### Cadastro
- Formulário completo (dados pessoais, características físicas, perfil operacional)
- Upload de foto de rosto + fotos extras com preview
- Tags para locais frequentes e substâncias
- Retreinamento automático após salvar

### Busca
- Full-text em nome, apelido, endereço
- Filtros por risco e status
- Paginação (20 por página)
- Ficha completa com histórico de ocorrências

### Ficha da Pessoa
- Todos os dados cadastrais
- Histórico de ocorrências com data, tipo, local
- Edição rápida de risco e status
- Upload de nova foto (retreina automaticamente)
- Exclusão com confirmação
