# Ambiente do minicurso: TT-Metalium + simulador ttsim (Wormhole).
#
# Nada é compilado na construção da imagem: o repositório é montado como
# volume em /workdir pelo docker-compose.yml.
#
# ATENÇÃO: o código exige TT-Metalium v0.77.0. A tag publicada `latest-rc`
# (= v1.1.0) desta imagem ainda traz o tt-metal v0.68.0-dev; veja a seção
# "Ambiente" do README. Use BASE_IMAGE para apontar para uma imagem v0.77.0
# (por exemplo, uma construída com petro-tt/Dockerfile.build).
ARG BASE_IMAGE=ghcr.io/cesar-hpc/petro-tt/cesar-wormhole-simulator-ubuntu-22.04-release-amd64:latest-rc
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

CMD ["tail", "-f", "/dev/null"]
