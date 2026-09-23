# Container image

The extraction pipeline runs as a container on AWS Batch (Fargate, linux/amd64).

Built by `.github/workflows/container.yml` in this repository — where the
Dockerfile lives — and pushed to the ECR repository that the app repository's
`infra/` creates. It is not built locally: the development machine has no
Docker, Fargate wants linux/amd64, and a CI-built image is one fewer thing for a
judge to install. The workflow assumes an IAM role over GitHub's OIDC token
(repository variables `AWS_ROLE_ARN` and `ECR_REPOSITORY`); it skips itself
where those are unset, so a fork does not fail.

The role's trust condition matches the token's `sub` claim, and GitHub now
writes that claim with the owner's and repository's numeric ids —
`repo:saeedbashir@5606473/signed-track-manifest@1381415665:ref:refs/heads/main`
— so the pattern `repo:owner/name:*` from most examples no longer matches and
STS answers "Not authorized to perform sts:AssumeRoleWithWebIdentity" with no
further hint. CloudTrail's record of the failed call shows the exact `sub`;
the trust policy allows both forms.

On Batch the container has no local files. `stm run s3://bucket/sources.yaml`
fetches the list, `url: s3://…` on an entry fetches the video, and with
`STM_ASSETS_BUCKET` set (the job definition sets it) each finished title is
uploaded to `s3://$STM_ASSETS_BUCKET/titles/<id>/`, its manifest's URLs made
absolute under `STM_PUBLIC_BASE` if that is set. `--upload` and `--public-base`
do the same by hand.

To run a title locally instead, skip the container entirely:

```sh
stm run catalogue/sources.yaml --out out --work work
```

The image ships no model weights. YOLOX-Tiny is downloaded on first use into
`STM_MODEL_DIR` (`/tmp/models` in the container, on Fargate task storage) and
verified against the checksum in `THIRD-PARTY-MODELS.md`.
