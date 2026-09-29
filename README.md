# Index-Translate

Open-source applications built on the [Index-Translate](https://huggingface.co/collections/IndexTeam/index-translate) model family.

## Projects

- [`extension/`](extension/) — **index-page-mt**: a browser extension (Chrome / Edge / Firefox) that translates web pages with a locally deployed Index-Translate LLM via any OpenAI-compatible API.
- [`video-dub/`](video-dub/) — **index-dub**: an end-to-end video translation & dubbing pipeline. Feed in an mp4, get it back dubbed into another language by the Index S2ST speech-translation models (vocal separation → VAD segmentation → speech-to-speech translation → timeline-aligned re-muxing).
- [`inference/`](inference/) — minimal inference quickstarts & cases for every open-sourced model: text translation (Translate), long documents (Nailong), syllable-controlled output (Homura), speech-to-speech dubbing (Echo-S2ST) and speech-to-text subtitles (Echo-S2TT).

## License

Apache 2.0 — see [LICENSE](LICENSE).
