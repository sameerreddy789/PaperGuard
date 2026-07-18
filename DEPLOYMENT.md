# PaperGuard — Deployment Guide (Alibaba Cloud)

This covers containerizing PaperGuard and deploying it to Alibaba Cloud. The
Dockerfile and image have been **built and verified locally** (see
"Verification" below); the actual cloud deployment steps require an Alibaba
account and credentials that aren't available in this environment, so they're
documented here for you to run.

## What's already done and verified

- `Dockerfile` (repo root) — CPU-only PyTorch, non-root user, `$PORT`-driven,
  `HEALTHCHECK` against Streamlit's `/_stcore/health`.
- `.dockerignore` — excludes `training/`, `hf_cache/`, `.env`, logs, and
  training-only files from the build context.
- **Verified locally**: `docker build` succeeds (~830 MB final image, well
  under Alibaba FC's 30 GB container limit), the container starts, and both
  `GET /_stcore/health` and `GET /` return HTTP 200. Docker itself reported the
  running container as `(healthy)` via the built-in `HEALTHCHECK`.

## 1. Build and push the image

```powershell
# Build (CPU wheels by default; see the Dockerfile header for GPU builds).
docker build -t paperguard:latest .

# Log in to Alibaba Container Registry (ACR) — create a registry + namespace
# first in the ACR console if you haven't: https://ecs.console.aliyun.com -> ACR
docker login registry.<region>.aliyuncs.com --username <your-aliyun-account>

# Tag and push.
docker tag paperguard:latest registry.<region>.aliyuncs.com/<namespace>/paperguard:latest
docker push registry.<region>.aliyuncs.com/<namespace>/paperguard:latest
```

Replace `<region>` (e.g. `cn-hangzhou`, `ap-southeast-1`) and `<namespace>`
with your ACR instance's values.

## 2. Configure environment variables (do NOT bake secrets into the image)

Whichever compute option you pick below, set these as platform-level
environment variables/secrets — never commit them or put them in the
Dockerfile:

| Variable | Required | Notes |
|---|---|---|
| `DASHSCOPE_API_KEY` | For Qwen | Alibaba-native LLM backend (see `.env.example`) |
| `PAPERGUARD_LLM_PROVIDER` | optional | `dashscope` to force Qwen for sub-agent calls |
| `PAPERGUARD_CREW_MODEL` | optional | e.g. `dashscope/qwen-plus` for the crew LLM |
| `PAPERGUARD_CREW_API_BASE` | optional | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `PAPERGUARD_DASHSCOPE_API_BASE` | optional | same endpoint, for the sub-agent Qwen client |
| `CROSSREF_EMAIL` | optional | polite-pool CrossRef + OpenAlex rate limits (reused by both) |
| `OPENALEX_EMAIL` | optional | overrides `CROSSREF_EMAIL` for OpenAlex's polite pool specifically |
| `PAPERGUARD_DETECTOR_MODEL` | optional | defaults to the HF repo; override for a local/alt model |
| `PORT` | no | defaults to `8000`; the platform usually sets this itself |

The full list (with defaults/comments) is in `.env.example`.

## 3a. Option A — Function Compute 3.0 (recommended: serverless, scale-to-zero)

FC 3.0 supports custom containers up to 30 GB, GPU instances, and HTTP
triggers — a good fit for a low/bursty-traffic Streamlit app.

1. Console: **Function Compute** → **Create Function** → **Custom Container**.
2. Point it at the pushed image: `registry.<region>.aliyuncs.com/<namespace>/paperguard:latest`.
3. Set **Port** to `8000` (matching the Dockerfile's `EXPOSE`/`$PORT` default).
4. Instance spec: start with 1–2 vCPU / 2–4 GB RAM (CPU-only detector
   inference on a single paragraph at a time is not compute-heavy; scale up if
   you see cold-start or latency issues). No GPU needed for the CPU build.
5. Add the environment variables from step 2 under **Environment Variables**.
6. Set **Request timeout** to at least 180s (a full multi-agent analysis with
   the LLM crew can take 1–3 minutes, per the app's own spinner message).
7. Enable an **HTTP Trigger** (or bind a custom domain) to get a public URL.
8. Deploy, then open the trigger URL — you should see the PaperGuard UI.

CLI equivalent (if you prefer `aliyun` CLI / Serverless Devs over the console):
```bash
# Using Alibaba's Serverless Devs tool (s.yaml would reference the image above).
s deploy
```
(Exact `s.yaml` shape depends on your FC 3.0 setup; the console path above is
the more direct/beginner-friendly route the first time.)

## 3b. Option B — PAI-EAS (one-click model serving)

If you want managed autoscaling with more knobs (and are comfortable with the
PAI ecosystem), PAI-EAS also accepts custom containers:

1. Console: **PAI** → **EAS (Elastic Algorithm Service)** → **Deploy Service**
   → **Custom deployment (Container)**.
2. Image: same ACR path as above. Port: `8000`.
3. Instance type: a CPU instance (e.g. `ecs.c6.xlarge` or similar) is enough
   for the CPU-only detector; pick a GPU instance only if you rebuild the
   image with `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121`.
4. Set the same environment variables as step 2.
5. Deploy — PAI-EAS gives you a service URL directly.

## 3c. Option C — Plain ECS + Docker (simplest fallback)

If FC/PAI setup is more than you need for a demo:

```bash
# On a fresh Alibaba ECS instance (Ubuntu, Docker installed):
docker pull registry.<region>.aliyuncs.com/<namespace>/paperguard:latest
docker run -d --name paperguard -p 80:8000 \
  -e DASHSCOPE_API_KEY=... \
  -e PAPERGUARD_LLM_PROVIDER=dashscope \
  registry.<region>.aliyuncs.com/<namespace>/paperguard:latest
```
Open a Security Group rule for port 80 (or whichever host port you map), then
browse to the instance's public IP.

## 4. Live smoke-test checklist (do this after any deploy)

- [ ] `GET /_stcore/health` on the deployed URL returns `ok` (same check the
      Dockerfile's `HEALTHCHECK` already validated locally).
- [ ] Upload a small `.txt`/`.md` paper first (fastest path — skips PDF
      parsing) and confirm the Integrity Dashboard renders headline numbers.
- [ ] Upload a real PDF and confirm the AI Heatmap / Overlay tabs populate,
      and the "Download annotated PDF" button produces a highlighted PDF.
- [ ] Toggle the CrewAI crew on/off (sidebar) and confirm both paths work —
      the engine-path fallback should still produce a full report even if
      the LLM key/provider is misconfigured.
- [ ] Check citation checks resolve against the live CrossRef/Semantic
      Scholar APIs (outbound internet access must be allowed from the
      compute environment).

## Notes / honest limitations

- **Cold starts**: FC's scale-to-zero means the first request after idle will
  be slow (loading torch + the deberta-v3-large detector model, ~1.7GB --
  noticeably heavier than the previous DistilBERT's 260MB, since the model was
  swapped to `desklib/ai-text-detector-v1.01` for its much higher disguised-AI
  recall; see `PROJECT_REPORT.md` Section 1). If demo timing matters, either
  keep a warm instance (PAI-EAS or ECS) or set FC's minimum instances ≥ 1 for
  the demo window.
- **GPU is optional, not required**: the detector runs one short paragraph at
  a time; CPU inference is fine for interactive use. Only reach for a GPU
  instance if you're doing something batch-heavy that this app doesn't do.
- **This guide was verified up through local Docker build/run only** — the
  actual Alibaba account setup, ACR push, and live FC/PAI deployment steps
  above have not been executed in this session (no cloud credentials
  available here). Follow them yourself, or share credentials if you want
  this done end-to-end in a future session.
