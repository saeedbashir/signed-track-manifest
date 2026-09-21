# Container image

The extraction pipeline runs as a container on AWS Batch (Fargate, linux/amd64).

Built by `.github/workflows/container.yml` in the app repository and pushed to
the ECR repository that `infra/` creates. It is not built locally: the
development machine has no Docker, and a CI-built image is one fewer thing for
a judge to install.

To run a title locally instead, skip the container entirely:

```sh
stm run catalogue/sources.yaml --out out --work work
```

The image ships no model weights. YOLOX-Tiny is downloaded on first use into
`STM_MODEL_DIR` (`/tmp/models` in the container, on Fargate task storage) and
verified against the checksum in `THIRD-PARTY-MODELS.md`.
