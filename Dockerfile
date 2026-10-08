# Ambiente do minicurso: TT-Metalium v0.77.0 + simulador ttsim (Wormhole).
#
# Nada é compilado na construção da imagem: o repositório é montado como
# volume em /workdir pelo docker-compose.yml.
ARG BASE_IMAGE=ghcr.io/cesar-hpc/petro-tt/cesar-tt-metalium-ubuntu-22.04-release-amd64:v.2.0.0
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

CMD ["tail", "-f", "/dev/null"]
