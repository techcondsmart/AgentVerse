# 🎬 Guia passo a passo: como analisar um vídeo com o swarm (para leigos)

Este guia ensina, do zero, como entregar um vídeo para o sistema e receber de
volta o **conteúdo integral analisado**: quadros-chave (imagens), textos que
aparecem na tela, transcrição do áudio (falas), descrições de cada cena com
horário exato, e — se você quiser — respostas verificadas por um time de
agentes de IA que confere cada afirmação contra as evidências (anti-alucinação).

Nenhum conhecimento de programação é necessário: é copiar e colar comandos.

---

## Parte 1 — O que você precisa (uma única vez)

### 1.1 Um computador com Python
- **Windows**: instale o Python em https://www.python.org/downloads/ (marque a
  opção *"Add Python to PATH"* durante a instalação).
- **Mac/Linux**: o Python geralmente já vem instalado. Confira abrindo o
  terminal e digitando `python3 --version`.

### 1.2 Uma chave do Google Gemini (gratuita)
1. Acesse https://aistudio.google.com/apikey
2. Entre com sua conta Google e clique em **"Create API key"**.
3. Copie a chave (começa com `AIzaSy...`) e guarde-a — você vai colá-la no
   passo 2.3.

> 💡 **Dica**: crie 2 ou 3 chaves (em projetos diferentes do Google Cloud).
> O sistema alterna entre elas automaticamente e pula as que falharem.
>
> ⚠️ **Privacidade**: no plano gratuito, o Google pode usar o conteúdo enviado
> para treinar seus modelos. **Não envie vídeos confidenciais** pelo plano
> gratuito — para isso, use o plano pago ou um modelo local (veja a Parte 5).

### 1.3 Baixar o sistema
Abra o terminal (no Windows: "Prompt de Comando" ou "PowerShell") e cole,
uma linha por vez:

```bash
git clone https://github.com/techcondsmart/AgentVerse.git
cd AgentVerse
git checkout claude/videoagent-repo-analysis-3ogm8w
```

> Se aparecer "git não encontrado", instale o Git em https://git-scm.com/downloads

### 1.4 Instalar os componentes
Ainda no terminal, dentro da pasta `AgentVerse`:

```bash
pip install -r requirements.txt
pip install -r agentverse/video/requirements-video.txt
pip install faster-whisper
```

- A primeira linha instala o sistema de agentes.
- A segunda instala a parte de vídeo (extração de quadros, detecção de cenas).
- A terceira instala o **Whisper**, que transcreve o áudio/falas do vídeo.
  (Opcional, mas recomendado — sem ele a análise é só visual.)

A instalação demora alguns minutos. Só precisa ser feita uma vez.

---

## Parte 2 — Preparar o vídeo e as chaves

### 2.1 Formatos aceitos
Qualquer formato comum de vídeo funciona: **MP4** (recomendado), MOV, AVI,
MKV, WEBM. Se o seu vídeo está no celular, transfira para o computador
(cabo, WhatsApp Web, Google Drive etc.).

### 2.2 Onde colocar o vídeo
Copie o arquivo para dentro da pasta `AgentVerse` (a mesma onde você está no
terminal). Exemplo: `AgentVerse/meu_video.mp4`.
*(Pode ficar em qualquer pasta — aí basta usar o caminho completo no comando.)*

### 2.3 Criar o arquivo de chaves (.env)
Dentro da pasta `AgentVerse`, crie um arquivo de texto chamado exatamente
`.env` (com o ponto na frente) contendo suas chaves, uma por linha:

```
GEMINI_API_KEY_1=AIzaSy...sua-chave-aqui
GEMINI_API_KEY_2=AIzaSy...outra-chave-se-tiver
```

> No Windows, use o Bloco de Notas e salve como `.env` (em "Tipo", escolha
> "Todos os arquivos" para não virar `.env.txt`).
>
> Se você já tem um `.env` maior (com pools de várias APIs), pode usá-lo
> direto — o sistema pega o que precisa e ignora o resto.

---

## Parte 3 — Analisar o vídeo (o passo principal)

### 3.1 Extração completa do conteúdo
No terminal, dentro da pasta `AgentVerse`:

```bash
python -m agentverse.video.analyze meu_video.mp4
```

Pronto. O sistema vai:
1. Detectar as **cenas** e extrair os **quadros-chave** (onde algo muda);
2. **Transcrever o áudio** (falas) com horário de cada trecho;
3. **Descrever cada quadro com IA**: objetos, ações, cenário e **qualquer
   texto que apareça na tela** (placas, legendas, slides);
4. Montar a **linha do tempo de evidências** com horário exato de tudo.

### 3.2 Onde ficam os resultados
Ao final, abra a pasta `analise_video/` criada ao lado do vídeo:

| Arquivo/pasta | O que contém |
|---|---|
| `frames/` | as **imagens** dos quadros-chave extraídos |
| `evidencias.md` | a **linha do tempo completa** — abra em qualquer editor: cada linha tem o horário `[mm:ss]`, o canal (VISÃO/ÁUDIO), a confiança e a descrição |
| `transcricao.txt` | **tudo que foi falado** no vídeo, com horários |
| `evidencias.json` | os mesmos dados em formato estruturado (para planilhas/programas) |

### 3.3 Fazer perguntas ao vídeo

```bash
python -m agentverse.video.analyze meu_video.mp4 --pergunta "Quem aparece no vídeo e o que ele diz?"
```

O sistema busca as evidências relevantes e as mostra com os horários.

### 3.4 Resposta verificada pelo swarm (máxima confiabilidade)

```bash
python -m agentverse.video.analyze meu_video.mp4 --pergunta "O que acontece no vídeo?" --swarm
```

Com `--swarm`, a resposta passa pelo **time de agentes verificadores**:
um agente responde citando os horários das evidências, três críticos
independentes tentam refutar cada afirmação (verificação em cadeia), e um
avaliador só aceita a resposta quando **100% das afirmações estão comprovadas**
pelas evidências. A resposta final fica em `analise_video/resposta.md`.

> É mais lento (alguns minutos) e gasta mais cota — use quando a precisão
> importa mais que a velocidade.

---

## Parte 4 — Ajustes úteis (opcionais)

| Opção | Para quê | Exemplo |
|---|---|---|
| `--saida PASTA` | mudar a pasta de resultados | `--saida resultado_reuniao` |
| `--env ARQUIVO` | usar um .env em outro lugar | `--env C:\chaves\.env` |
| `--max-frames N` | vídeos longos: limitar quadros analisados (controla cota/custo) | `--max-frames 100` |
| `--whisper TAM` | qualidade do áudio: `tiny`/`small`/`medium`/`large-v3` | `--whisper large-v3` |
| `--pausa SEG` | pausa entre chamadas de IA (cota gratuita) | `--pausa 7` |
| `--modelo NOME` | trocar o modelo de IA | `--modelo gemini-2.5-flash` |

**Guia rápido de duração** (com `--pausa 5` padrão):
- Vídeo de 1 min ≈ 5–15 quadros ≈ 2–4 min de análise
- Vídeo de 10 min ≈ 30–60 quadros ≈ 5–15 min
- Vídeo de 1 h: use `--max-frames 150` e paciência ☕

---

## Parte 5 — Problemas comuns e soluções

| Sintoma | Causa | Solução |
|---|---|---|
| `vídeo não encontrado` | nome/caminho errado | confira o nome exato; arraste o arquivo para o terminal para colar o caminho |
| `sem IA de visão` | .env não encontrado ou sem chave GEMINI | confira o passo 2.3 (arquivo `.env` na pasta onde você roda o comando, variável `GEMINI_API_KEY_1=`) |
| Erro `403 ... denied` | chave revogada/projeto bloqueado | gere outra chave; o sistema pula a ruim sozinho se houver outras |
| Erro `429 ... quota` | cota diária do plano grátis esgotou | espere até o dia seguinte, use outra chave, ou troque de modelo (`--modelo gemini-flash-lite-latest` tem cota maior) |
| `transcription unavailable` | Whisper não instalado | `pip install faster-whisper` |
| Análise muito lenta | vídeo grande | reduza `--max-frames`; aumente `--pausa` só se houver erros de cota |
| Vídeo confidencial | plano grátis treina nos dados | use plano pago do Gemini **ou** um modelo local: instale um servidor vLLM/Ollama com Qwen-VL e veja `documentation/video_models_reference.md` |

---

## Resumo ultra-rápido (cola)

```bash
# 1) uma vez só:
git clone https://github.com/techcondsmart/AgentVerse.git && cd AgentVerse
git checkout claude/videoagent-repo-analysis-3ogm8w
pip install -r requirements.txt -r agentverse/video/requirements-video.txt faster-whisper
#    crie o arquivo .env com:  GEMINI_API_KEY_1=AIzaSy...

# 2) sempre que quiser analisar um vídeo:
python -m agentverse.video.analyze meu_video.mp4

# 3) para perguntar com verificação anti-alucinação:
python -m agentverse.video.analyze meu_video.mp4 --pergunta "..." --swarm
```

*Como funciona por dentro: `documentation/video_understanding.md`.
Modelos/APIs gratuitos: `documentation/video_models_reference.md`.*
