# Forepaw Vercel replay

This directory is a static, public-safe replay package. Vercel serves `public/`,
which contains the retained dimOS-to-true-Go2 MjLab telemetry plus two direct
MjLab reference rehearsals. It does not run MuJoCo, dimOS, or the controller on
Vercel.

The included runs use the explicitly labeled privileged reference model and do
not constitute learned-world-model evidence.

Deploy from this directory with:

```bash
vercel --prod --yes
```
