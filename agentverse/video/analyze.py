"""Análise completa de um vídeo em um único comando — pensado para leigos.

Uso básico:
    python -m agentverse.video.analyze MEU_VIDEO.mp4

Com pergunta e swarm de verificação:
    python -m agentverse.video.analyze MEU_VIDEO.mp4 \
        --pergunta "O que acontece no vídeo?" --swarm

O que ele faz, na ordem:
  1. Carrega suas chaves de API de um arquivo .env (--env, padrão ./.env);
  2. Extrai os quadros-chave do vídeo (cortes de cena + movimento);
  3. Transcreve o áudio com Whisper (se instalado: pip install faster-whisper);
  4. Descreve cada quadro com IA de visão (Gemini), incluindo textos na tela;
  5. Monta o índice de evidências com timestamps;
  6. Salva tudo na pasta de saída (frames/, evidencias.md, transcricao.txt,
     evidencias.json);
  7. Se --pergunta for dada: busca as evidências relevantes e responde;
     com --swarm, a resposta passa pelo time de agentes verificadores
     (anti-alucinação, mais lento e gasta mais cota).

Tudo falha de forma suave: sem Whisper analisa só o visual; sem chave só
extrai frames; chaves ruins são puladas automaticamente (KeyPool).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


# ---------------------------------------------------------------- .env -----
def load_env_file(path: str) -> int:
    """Carrega variáveis de um .env simples (NOME=valor). Retorna quantas."""
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if value.strip():
            os.environ.setdefault(name.strip(), value.strip())
            n += 1
    return n


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agentverse.video.analyze",
        description="Extrai e analisa o conteúdo integral de um vídeo "
                    "(frames, textos na tela, áudio, evidências com tempo).",
    )
    ap.add_argument("video", help="caminho do arquivo de vídeo (mp4, mov, avi...)")
    ap.add_argument("--env", default=".env",
                    help="arquivo .env com as chaves (padrão: ./.env)")
    ap.add_argument("--saida", default="analise_video",
                    help="pasta onde salvar os resultados")
    ap.add_argument("--pergunta", default=None,
                    help="pergunta sobre o vídeo (opcional)")
    ap.add_argument("--swarm", action="store_true",
                    help="responde via time de agentes verificadores (anti-alucinação)")
    ap.add_argument("--whisper", default="small",
                    help="tamanho do modelo de áudio: tiny/small/medium/large-v3")
    ap.add_argument("--max-frames", type=int, default=40,
                    help="máximo de quadros analisados pela IA (controla custo/cota)")
    ap.add_argument("--modelo", default="gemini-flash-lite-latest",
                    help="modelo de visão/razão (padrão tem cota diária maior)")
    ap.add_argument("--pausa", type=float, default=5.0,
                    help="pausa em segundos entre chamadas de IA (respeita cota grátis)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"ERRO: vídeo não encontrado: {args.video}")
        return 2

    n_env = load_env_file(args.env)
    os.environ.setdefault("OPENAI_API_KEY", "sk-nao-usado")  # exigido pelo import
    print(f"[1/6] Chaves carregadas de {args.env}: {n_env} variáveis")

    from agentverse.video import SamplingConfig, VideoPerceptionPipeline
    from agentverse.video.backends import KeyPool, pooled_vision_backend

    BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
    backends = {}
    kp = None
    try:
        kp = KeyPool.from_env("GEMINI_API_KEY")
        raw = pooled_vision_backend(args.modelo, BASE, kp, max_tokens=2048)

        def paced(image_path, prompt):
            time.sleep(args.pausa)
            return raw(image_path, prompt)

        backends["gemini"] = paced
        print(f"[2/6] IA de visão pronta: {args.modelo} "
              f"({len(kp.healthy())} chave(s) no pool)")
    except Exception as e:
        print(f"[2/6] AVISO: sem IA de visão ({e}). Vou extrair frames e áudio "
              f"mesmo assim, sem descrições.")

    os.makedirs(args.saida, exist_ok=True)
    pipeline = VideoPerceptionPipeline(
        caption_backends=backends or {"nenhum": lambda p, q: "{}"},
        sampling=SamplingConfig(max_frames=args.max_frames),
        whisper_size=args.whisper,
    )
    print("[3/6] Analisando o vídeo (frames + áudio + descrições)... "
          "isso pode levar alguns minutos.")
    k = pipeline.process(args.video, args.saida)

    # ------------------------------------------------------------ salvar ----
    timeline = k.index.timeline()
    md = [f"# Análise do vídeo: {os.path.basename(args.video)}\n",
          f"- Quadros-chave analisados: {k.n_frames}",
          f"- Trechos de fala transcritos: {k.n_transcript_segments}",
          f"- Quadros com baixa confiança: {len(k.low_agreement_timestamps)}\n",
          "## Linha do tempo (evidências)\n"]
    for h in timeline:
        canal = "VISÃO" if h.modality == "vision" else "ÁUDIO"
        md.append(f"- **[{fmt_ts(h.timestamp)}] {canal}** "
                  f"(confiança {h.agreement:.0%}): {h.text}")
    open(os.path.join(args.saida, "evidencias.md"), "w", encoding="utf-8").write(
        "\n".join(md))

    with open(os.path.join(args.saida, "transcricao.txt"), "w", encoding="utf-8") as f:
        fala = [h for h in timeline if h.modality == "audio"]
        if fala:
            for h in fala:
                f.write(f"[{fmt_ts(h.timestamp)}] {h.text}\n")
        else:
            f.write("(sem áudio transcrito — instale: pip install faster-whisper)\n")

    json.dump(
        [{"tempo_s": h.timestamp, "canal": h.modality, "confianca": h.agreement,
          "texto": h.text, "fonte": h.source} for h in timeline],
        open(os.path.join(args.saida, "evidencias.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    print(f"[4/6] Resultados salvos em: {args.saida}/ "
          f"(frames/, evidencias.md, transcricao.txt, evidencias.json)")

    if not args.pergunta:
        print("[5/6] Nenhuma --pergunta informada; análise concluída.")
        print("[6/6] Dica: rode de novo com --pergunta \"...\" para perguntar ao vídeo.")
        return 0

    # ------------------------------------------------- responder pergunta ----
    hits = k.index.query(args.pergunta, k=10)
    if not hits:
        # Busca lexical não casou (ex.: pergunta em português, descrições em
        # inglês). Para vídeos curtos, a linha do tempo inteira é a melhor
        # evidência possível — use-a em vez de responder "nada encontrado".
        hits = timeline
    evid = "\n".join(
        f"[{h.timestamp:0.1f}s | {h.modality} | agreement={h.agreement:.2f}] {h.text}"
        for h in hits) or "(nenhuma evidência encontrada)"

    if not args.swarm:
        print("[5/6] Evidências mais relevantes para a pergunta:")
        print(evid)
        print("[6/6] Para uma resposta verificada por agentes, adicione --swarm.")
        return 0

    print("[5/6] Acionando o swarm de verificação (isso usa mais cota)...")
    from agentverse.llms.openai import OpenAIChat
    from agentverse.tasksolving import TaskSolving

    _gen, _agen = OpenAIChat.generate_response, OpenAIChat.agenerate_response

    def _pg(self, *a, **kw):
        time.sleep(args.pausa)
        return _gen(self, *a, **kw)

    async def _pa(self, *a, **kw):
        time.sleep(args.pausa)
        return await _agen(self, *a, **kw)

    OpenAIChat.generate_response, OpenAIChat.agenerate_response = _pg, _pa

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts = TaskSolving.from_task("video_understanding",
                               os.path.join(here, "tasks", "tasksolving"))
    env = ts.environment
    env.task_description = (args.pergunta +
                            "\n\n# Retrieved evidence from the video\n" + evid)
    env.max_turn = 3
    env.rule.executor.index = k.index
    env.rule.add_execution_result_to_critic = True
    env.rule.add_execution_result_to_solver = True

    plain = ("\n\nIMPORTANT: respond in PLAIN TEXT only — no markdown, no bold, "
             "no code fences. Follow the response format EXACTLY as specified.")
    import itertools

    # Reusa o pool da percepção: chaves já marcadas como ruins são puladas.
    swarm_pool = kp if kp is not None else KeyPool.from_env("GEMINI_API_KEY")
    keys = itertools.cycle(swarm_pool.healthy())
    for _, agent_or_list in env.agents.items():
        for a in (agent_or_list if isinstance(agent_or_list, list)
                  else [agent_or_list]):
            a.llm.client_args = {"api_key": next(keys), "base_url": BASE}
            a.llm.is_azure = False
            a.llm.args.model = args.modelo
            a.llm.args.max_tokens = 2048
            a.llm.args.temperature = 0
            a.max_retry = 3
            a.append_prompt_template = (a.append_prompt_template or "") + plain

    plan, result, logs = ts.run()
    resposta = plan or "(o swarm não chegou a uma resposta aceita)"
    open(os.path.join(args.saida, "resposta.md"), "w", encoding="utf-8").write(
        f"# Pergunta\n{args.pergunta}\n\n# Resposta verificada\n{resposta}\n")
    print("[6/6] RESPOSTA VERIFICADA PELO SWARM:")
    print(resposta)
    print(f"\n(salva em {args.saida}/resposta.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
