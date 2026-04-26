# FaceWatch 🔍

Sistema web de identificação e cadastro de pessoas para uso operacional por agentes de segurança.
Acessa via browser — desktop e celular na mesma rede Wi-Fi via HTTPS.

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Backend | Python 3.10+, Flask, Flask-SQLAlchemy (SQLite), Flask-Bcrypt |
| Reconhecimento principal | FaceNet (InceptionResnetV1 VGGFace2) + MTCNN via facenet-pytorch |
| Reconhecimento fallback | OpenCV HOG + KNN (se facenet-pytorch não estiver instalado) |
| Frontend | HTML/CSS/JS puro — sem frameworks externos |
| SSL | Certificado auto-assinado gerado automaticamente via OpenSSL ou lib cryptography |
| Servidor | Flask dev server com HTTPS, porta 5443 |

---

## Estrutura do Projeto

```
facewatch/
├── run.py                        # Ponto de entrada — inicia o servidor HTTPS
├── import_faces.py               # Importador em lote via linha de comando
├── requirements.txt              # Dependências pip (sem torch — instalar via conda)
├── install.sh                    # Script de instalação Linux/macOS
├── README.md
│
├── app/
│   ├── __init__.py               # App factory — cria app, migra DB, semeia usuários
│   ├── models.py                 # Models SQLAlchemy: User, Person, Occurrence
│   ├── routes/
│   │   ├── auth.py               # Endpoints de autenticação
│   │   ├── api.py                # Todos os endpoints JSON da API
│   │   └── pages.py              # Rotas que retornam HTML (render_template)
│   └── services/
│       └── face.py               # Motor de reconhecimento facial completo
│
├── templates/
│   ├── base.html                 # Layout base, CSS global, JS utilitários compartilhados
│   ├── login.html                # Tela de login
│   ├── identify.html             # Scanner em tempo real com câmera
│   ├── register.html             # Cadastro de pessoa (3-5 fotos obrigatórias)
│   ├── search.html               # Busca paginada com filtros
│   ├── person.html               # Ficha completa + ocorrências + gestão de fotos
│   └── import.html               # Importação em lote via browser
│
├── static/
│   └── uploads/
│       ├── faces/                # Fotos de rosto (usadas no treino do modelo)
│       └── extras/               # Fotos extras: corpo, tatuagens, documentos
│
├── database/
│   ├── facewatch.db              # Banco SQLite — criado automaticamente
│   └── knn_model.pkl             # Modelo serializado (FaceNet embeddings ou HOG+KNN)
│
└── instance/
    ├── cert.pem                  # Certificado SSL — gerado automaticamente
    └── key.pem                   # Chave privada SSL
```

---

## Instalação

### ⚠️ Windows — Use Conda (único método testado e funcional)

O pip puro no Windows causa conflitos de build com PyTorch e Pillow.
**Use Conda obrigatoriamente no Windows.**

```powershell
# 1. Criar ambiente isolado
conda create -n face python=3.10
conda activate face

# 2. Instalar PyTorch CPU via canal oficial conda
conda install pytorch torchvision cpuonly -c pytorch

# 3. Instalar facenet-pytorch e demais dependências via pip
pip install facenet-pytorch
pip install flask flask-sqlalchemy flask-bcrypt
pip install opencv-python-headless pillow scikit-learn numpy

# 4. Rodar
cd facewatch
python run.py
```

> ❌ **Não faça** `pip install torch` direto sem conda no Windows.
> Vai falhar com erros de build do Pillow ou falta de espaço em disco.

> ❌ **Não use** `pip install -r requirements.txt` para instalar torch no Windows.
> O requirements.txt é para as demais dependências apenas.

### Linux / macOS

```bash
# Opção 1: script automático
chmod +x install.sh && ./install.sh
source venv/bin/activate
python run.py

# Opção 2: manual
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch
pip install flask flask-sqlalchemy flask-bcrypt opencv-python-headless pillow scikit-learn numpy
python run.py
```

### Verificar se FaceNet instalou corretamente

```powershell
python -c "
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
model = InceptionResnetV1(pretrained='vggface2').eval()
out = model(torch.randn(1, 3, 160, 160))
print('FaceNet OK — embedding dim:', out.shape[1])
"
```

Resultado esperado: `FaceNet OK — embedding dim: 512`

Se falhar, o sistema usa HOG+KNN automaticamente (qualidade inferior).
O terminal mostra qual motor está ativo:
```
[TREINO] FaceNet (VGGFace2)    ← motor correto
[TREINO] HOG+KNN (fallback)    ← facenet-pytorch não instalado
```

---

## Iniciar o Servidor

```powershell
conda activate face
cd C:\Users\vitor\Desktop\facewatch\facewatch
python run.py
```

Na primeira execução:
- Gera certificado SSL automaticamente (`instance/cert.pem` e `instance/key.pem`)
- Cria o banco de dados SQLite (`database/facewatch.db`)
- Cria usuários padrão (admin e policial)
- Imprime os endereços de acesso no terminal

---

## Acesso

```
Desktop  →  https://localhost:5443
Mobile   →  https://192.168.X.X:5443   (IP impresso no terminal)
```

> ⚠️ O browser exibe aviso de certificado não confiável — é normal.
> Chrome: clique em **Avançado → Prosseguir para localhost**
> Firefox: clique em **Aceitar o risco e continuar**
> Safari / iOS: vá em Ajustes → Geral → VPN e Gerenciamento de Dispositivos → confie no certificado

> 📱 A câmera (`getUserMedia`) **só funciona em HTTPS**. Por isso o sistema roda
> obrigatoriamente em HTTPS mesmo localmente — necessário para o celular funcionar.

---

## Usuários Padrão

Criados automaticamente na primeira execução:

| Usuário | Senha | Papel |
|---------|-------|-------|
| admin | admin123 | Administrador |
| policial | policial123 | Agente |

---

## Motor de Reconhecimento Facial

### FaceNet — Motor Principal

```
Imagem recebida (câmera / upload / base64)
    ↓
PIL.ImageOps.exif_transpose()
    Corrige rotação EXIF — fotos de celular chegam rotacionadas
    ↓
MTCNN (Multi-task Cascaded CNN)
    Detecta rosto, alinha pelos olhos, recorta 160×160px
    Modo realtime  → rejeita se não detectar rosto (evita falsos positivos)
    Modo cadastro  → aceita fallback crop central se MTCNN falhar
    ↓
InceptionResnetV1 — pré-treinado em VGGFace2
    Gera embedding de 512 dimensões
    Rede treinada em 3.3 milhões de rostos de 9.131 pessoas
    ↓
Similaridade Cosseno
    Compara o embedding com todos os cadastrados no banco
    Retorna a pessoa com menor distância
    ↓
Threshold calibrado automaticamente
    Aceita se distância < threshold E engine confia
```

### Calibração do Threshold

Calculado automaticamente a cada retreino com base nos dados reais:

```
Com múltiplas pessoas e múltiplas fotos:
  threshold = ponto médio entre (pior intra-classe) e (melhor inter-classe)

Com múltiplas pessoas e 1 foto cada:
  threshold = 80% da menor distância inter-classe

Com 1 pessoa e múltiplas fotos:
  threshold = pior variação intra-classe × 1.5

Com 1 pessoa e 1 foto:
  threshold = 0.90 (fixo conservador)
```

O terminal mostra os valores após cada retreino:
```
[TREINO] Threshold=0.84 (intra_max=0.71 inter_min=0.98)
[IDENTIFY] dist=0.63 threshold=0.84 pid=1
[IDENTIFY] → identified pid=1 conf=87.2%
```

### HOG+KNN — Fallback

Usado quando `facenet-pytorch` não está disponível.

```
Imagem → Haar Cascade (detecta rosto) → CLAHE → HOG 64×64 → KNN → Threshold
```

Limitações do HOG+KNN:
- Sensível a mudanças de ângulo, iluminação e distância
- Pode confundir rostos similares
- Threshold fixo menos robusto

---

## Boas Práticas de Cadastro

**O reconhecimento melhora diretamente com a qualidade das fotos de cadastro.**

### Quantidade
- Mínimo: **3 fotos** (obrigatório para salvar)
- Recomendado: **5 fotos**

### Variedade recomendada
- Frontal — olhando direto para a câmera
- Levemente de lado esquerdo (~15°)
- Levemente de lado direito (~15°)
- Com óculos / sem óculos (se usar)
- Em iluminação diferente (luz natural e artificial)

### O que evitar
- Fotos com rosto coberto ou desfocado
- Múltiplas fotos idênticas (mesma pose, mesmo momento)
- Fotos muito escuras ou superexpostas
- Rosto muito pequeno na imagem (use a câmera mais próxima)

---

## Funcionalidades

### 🔍 Identificar
- Câmera frontal e traseira — botão de troca durante o scan
- Scan automático a cada 1.5s com HUD overlay no vídeo
- Detecta rosto com MTCNN antes de consultar o modelo (evita falsos positivos)
- Botão de captura para frame completo em alta resolução
- Upload de arquivo para identificação offline
- Painel de resultado: foto, nome, risco, status, confiança, link para perfil
- Botão de registrar ocorrência direto do resultado

### 👤 Cadastro
- Formulário completo: nome, apelido, gênero, idade, cor de pele, altura, endereço
- Tags para locais frequentes e substâncias associadas
- Campos para tatuagens, marcas físicas, observações
- Nível de risco (baixo / médio / alto) e status (ativo / detido / foragido / liberado)
- 5 slots visuais de foto de rosto — câmera inline ou upload de arquivo
- Mínimo de 3 fotos obrigatório para salvar
- Fotos extras: corpo, tatuagens, documentos
- Retreino automático em background após salvar

### 🔎 Busca
- Full-text em nome, apelido e endereço
- Filtros por nível de risco e status
- Paginação (20 por página)
- Click abre a ficha completa

### 📋 Ficha da Pessoa
- Todos os dados cadastrais
- Grade de fotos de rosto com add/remove individual (máx 5)
- Aviso quando tem menos de 3 fotos
- Histórico de ocorrências com data, tipo, local e agente responsável
- Edição rápida de risco e status
- Exclusão com confirmação

### 📦 Importação em Lote
- Via browser: arrasta imagens, preview, importa com barra de progresso
- Via CLI: `python import_faces.py --pasta C:\fotos`
- Nome do arquivo vira nome da pessoa (`joao_silva.jpg` → "Joao Silva")
- Detecta duplicatas por nome e adiciona como foto extra
- Formatos: JPG, PNG, WEBP, BMP, TIFF

---

## Endpoints da API

```
# Autenticação
POST   /api/login                           { username, password }
POST   /api/logout
GET    /api/me

# Pessoas
GET    /api/stats
GET    /api/persons                         ?q= &risk= &status= &page= &per_page=
POST   /api/persons                         { name, nickname, gender, age, skin_color,
                                              height_cm, address, frequent_places[],
                                              substances[], tattoos, physical_marks,
                                              observations, risk_level, status }
GET    /api/persons/<id>
PUT    /api/persons/<id>
DELETE /api/persons/<id>

# Fotos
POST   /api/persons/<id>/photo/face         { image_b64 }  → adiciona ao array (máx 5)
DELETE /api/persons/<id>/photo/face/<idx>   → remove por índice (0-based)
POST   /api/persons/<id>/photo/extra        { image_b64 }

# Ocorrências
POST   /api/persons/<id>/occurrences        { type, location, description, substances[] }

# Reconhecimento
POST   /api/recognize                       { image_b64 }  → identificação completa
POST   /api/recognize/realtime             { image_b64 }  → rápido, exige rosto detectado
POST   /api/retrain                         → força retreino completo

# Arquivos
GET    /api/uploads/<path>                  → serve fotos salvas
```

### Resposta do reconhecimento

```json
{
  "status": "identified",   // identified | unknown | no_face | no_model | feature_error
  "person_id": 3,
  "confidence": 87.2,       // 0-100
  "distance": 0.634,        // distância cosseno (FaceNet) ou euclidiana (HOG)
  "person": { ... }         // objeto Person completo (só quando identified)
}
```

---

## Forçar Retreino Manualmente

Via console do browser (F12):
```javascript
fetch('/api/retrain', {method:'POST', credentials:'same-origin'})
  .then(r => r.json())
  .then(console.log)
```

Via PowerShell (precisa de sessão autenticada via Postman ou similar):
```powershell
# 1. Login
Invoke-WebRequest -Uri "https://localhost:5443/api/login" -Method POST `
  -ContentType "application/json" `
  -Body '{"username":"admin","password":"admin123"}' -SessionVariable s

# 2. Retrain
Invoke-WebRequest -Uri "https://localhost:5443/api/retrain" -Method POST -WebSession $s
```

---

## Importador em Lote (CLI)

```powershell
# Simulação — não grava nada
python import_faces.py --pasta C:\fotos --dry-run

# Importação real com opções padrão
python import_faces.py --pasta C:\fotos

# Com risco e status específicos
python import_faces.py --pasta C:\fotos --risco high --status fugitive

# Ajuda
python import_faces.py --help
```

Convenção de nome de arquivo:
```
joao_silva.jpg    →  "Joao Silva"
MARIA-SOUZA.png   →  "Maria Souza"
pedro santos.webp →  "Pedro Santos"
```

---

## Solução de Problemas

### "Câmera não disponível"
- O sistema requer HTTPS — acesse por `https://` e não `http://`
- No celular, confirme que está na mesma rede Wi-Fi que o servidor
- Permita o acesso à câmera quando o browser solicitar

### "Não autenticado" na API
- A API usa sessão por cookie — faça login pelo browser ou pelo Postman com cookie persistente
- Sessões expiram ao reiniciar o servidor

### FaceNet não identifica a pessoa
1. Confirme que o terminal mostra `[TREINO] FaceNet (VGGFace2)` e não `HOG+KNN`
2. Force um retreino pelo console do browser
3. Verifique no terminal o threshold e a distância: `[IDENTIFY] dist=X threshold=Y`
4. Se `dist > threshold`, adicione mais fotos variadas da pessoa e retreine

### Certificado SSL rejeitado no iOS / Safari
- Vá em **Ajustes → Geral → VPN e Gerenciamento de Dispositivos**
- Encontre o certificado FaceWatch e toque em **Confiar**

### Erro ao instalar torch no Windows
- Use **conda** obrigatoriamente: `conda install pytorch torchvision cpuonly -c pytorch`
- Não use `pip install torch` diretamente no Windows

### Banco de dados corrompido ou desatualizado
- Delete `database/facewatch.db` e `database/knn_model.pkl`
- O banco é recriado automaticamente ao iniciar
- Recadastre as pessoas e retreine o modelo

---

## Segurança

- Autenticação por sessão Flask com senha hasheada (bcrypt)
- HTTPS obrigatório — câmera não funciona sem ele
- Certificado auto-assinado — adequado para uso em rede local fechada
- Para produção/hospedagem: substitua por certificado Let's Encrypt e configure proxy reverso (nginx)
- Não exponha a porta 5443 diretamente para a internet sem autenticação adicional

---

## Dados Técnicos do Modelo

| Parâmetro | Valor |
|-----------|-------|
| Arquitetura | InceptionResnetV1 |
| Dataset de pré-treino | VGGFace2 |
| Pessoas no pré-treino | 9.131 identidades |
| Fotos no pré-treino | 3.3 milhões |
| Dimensão do embedding | 512 |
| Métrica de comparação | Distância Cosseno |
| Detecção de rosto | MTCNN (3 estágios) |
| Threshold | Calibrado automaticamente por retreino |
| Serialização | pickle (.pkl) |
| Retreino incremental | Cache de embeddings por mtime do arquivo |
| Retreino em background | Thread daemon — não bloqueia a resposta HTTP |