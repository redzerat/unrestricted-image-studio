# Hardening plan for redzerat/unrestricted-image-studio fork
Date: 2026-11-25. Based on Muse's line-by-line read of upstream flux_app.py (503 lines, commit bffd9c9). No upstream code executed.

## Patches to apply on the fork (branch: hardening)

1. Auth fail-closed (web() / check())
   - If AUTH_TOKEN env is missing or empty, every authenticated endpoint returns 503 instead of matching an empty header.
   - Compare with secrets.compare_digest (constant time).
   - Keep /health unauthenticated but return only {"ok": true} (already the case).

2. Chinese-prompt translation OFF by default
   - translate_zh() only runs when env TRANSLATE_ZH=1 is explicitly set on the web function.
   - When off and prompt contains Chinese, return prompt_used unchanged and translated=false, plus a warning field noting translation is disabled for privacy.
   - Rationale: upstream sent Chinese prompts in plaintext to Google gtx and MyMemory.

3. Prompt / request limits
   - Reject prompts > 2000 chars (413/400).
   - Keep existing clamps: width/height 256-1536, steps <=60, guidance <=8, model/face whitelists.

4. Cost guardrails
   - Add Modal function max_containers / concurrency limit on Generator if supported by the installed Modal version at deploy time (verify syntax at deploy, do not guess).
   - Document in README-hardening: set a Modal workspace budget alert; token leak = GPU bill.

5. Supply-chain pinning (to verify at apply time, not guessed here)
   - Pin pip packages to the exact versions resolved in a test venv at apply time; record them in the commit message.
   - Pin Hugging Face model repos by revision (commit SHA) for: FLUX.2-klein-4B, abliterated text encoder, Realistic Vision, sd-vae-ft-mse, LUSTIFY-v2.0, Chroma1-HD, ponyRealism, IP-Adapter-FaceID. SHAs must be read from HF at apply time, never invented.

6. FaceID use boundary (documentation, not code)
   - README-hardening note: reference faces only for the user themself or a consenting adult; no third-party faces.

## Not changed
- Engine recipes, samplers, steps/CFG, lazy loading, A10G choice, scaledown 120s, timeout 1200s.
- No deployment to Modal in this step. Deployment is a separate user decision after the hardened fork is verified.

## Verification after push
- Read back flux_app.py and README from redzerat/unrestricted-image-studio via the GitHub connector and confirm the patched check()/translate gating text is present.
- Python syntax check is NOT done by importing modal code locally against the cloned upstream (execution-risk notice); syntax will be checked on the fork's own files in a clean venv only if the user approves deployment prep.
