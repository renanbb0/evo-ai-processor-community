# Fontes da instalação Bee

Recuperação em 21/10/2026 da imagem
`evoapicloud/evo-ai-processor-community:latest`, ID imutável
`sha256:1a97c7bc033658b7dcaacdd0ef10023521490b3a5c80870f60d9a9e7ed09521c`.
Não usar `latest` como identidade de uma instalação futura.

A imagem recebe cinco arquivos por bind mount, todos sob
`/opt/evocrm/processor-patch/`. A branch `bee/producao` salva os mesmos fontes
nas posições definitivas de `src/`:

| Arquivo no volume | Fonte salvo |
|---|---|
| `runner_utils.py` | `src/services/adk/runners/runner_utils.py` |
| `standard_runner.py` | `src/services/adk/runners/standard_runner.py` |
| `audio_transcription_service.py` | `src/services/audio_transcription_service.py` |
| `llm_model_routing.py` | `src/utils/llm_model_routing.py` |
| `llm_agent_builder.py` | `src/services/adk/agents/llm_agent_builder.py` |

Os três primeiros já existiam localmente sem commit; os outros dois foram
recuperados dos volumes, incluindo a ligação reversa entre agentes e o ajuste
de roteamento dos modelos. `source-audit.json` registra os hashes da imagem e
dos arquivos efetivamente montados. Nenhum volume de produção foi modificado.

O repositório `renanbb0/evo-ai-processor-community` existente é público.
Ele não atende à exigência D30 de cópia privada. Criar um destino privado e
enviar a branch exige o OK do usuário. Não houve push nesta recuperação.
