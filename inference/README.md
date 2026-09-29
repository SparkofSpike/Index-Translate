# inference · Index-Translate Model Quickstarts

[中文](README_zh.md)

Minimal, tested inference scripts and cases for the open-sourced
[Index-Translate model family](https://huggingface.co/collections/IndexTeam/index-translate).

## Model map

| Model | Task | Quickstart |
|---|---|---|
| [Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) / [9B](https://huggingface.co/IndexTeam/Index-Translate-9B) | general text translation (zh/en/ja + 20 more) | [`llm/`](llm/) |
| [Index-Nailong-2B](https://huggingface.co/IndexTeam/Index-Nailong-2B) / [9B](https://huggingface.co/IndexTeam/Index-Nailong-9B) | long-document translation (262K context, zh↔en / zh↔ja) | [`llm/`](llm/) |
| [Index-Homura-2B](https://huggingface.co/IndexTeam/Index-Homura-2B) / [9B](https://huggingface.co/IndexTeam/Index-Homura-9B) | syllable-controlled translation (fits dubbing time slots) | [`llm/`](llm/) |
| [Index-Echo-S2ST-2B](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B) / [9B](https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B) | speech-to-speech dubbing (zh/en source → en/es/ja/zh, voice-preserving) | [`echo-s2st/`](echo-s2st/) |
| [Index-Echo-S2TT-2B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B) / [9B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-9B) | speech-to-text translation (zh audio/video → en/ja/es subtitles) | [`echo-s2tt/`](echo-s2tt/) |

## TL;DR

```bash
# --- any text-translation LLM (Translate / Nailong / Homura) ---
pip install -U vllm            # need a build with Qwen3.5 support
./llm/serve_vllm.sh translate-9b
python llm/translate.py "你好，世界" --target en

# --- speech-to-speech dubbing ---
huggingface-cli download IndexTeam/Index-Echo-S2ST-2B --local-dir ./Index-Echo-S2ST-2B
python echo-s2st/dub.py input.wav --lang en -o dub_en.wav

# --- speech-to-text subtitles ---
python echo-s2tt/s2tt.py input.mp4 --lang en   # auto-downloads the 2B package
```

All scripts were tested end-to-end against the released weights before
publication; see each directory's README for details and `llm/cases/` for
input/output examples captured from real runs.

Related: [`../video-dub/`](../video-dub/) turns these models into a full
video-dubbing pipeline (vocal separation, VAD segmentation, timeline re-mux).
