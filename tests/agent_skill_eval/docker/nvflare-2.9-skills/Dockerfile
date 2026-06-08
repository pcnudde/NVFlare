FROM nvflare-agent-eval:basic

ARG BUILD_DATE=unknown
ARG NVFLARE_AGENT_SOURCE_REF=flare_agent

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${NVFLARE_AGENT_SOURCE_REF}" \
      org.opencontainers.image.source="https://github.com/chesterxgchen/NVFlare/tree/flare_agent" \
      nvflare.agent_eval.image="2.9-skills" \
      nvflare.agent_eval.skills="codex,claude"

RUN python -m pip install "git+https://github.com/chesterxgchen/NVFlare.git@${NVFLARE_AGENT_SOURCE_REF}" \
    && nvflare agent skills install --agent codex \
    && nvflare agent skills install --agent claude
