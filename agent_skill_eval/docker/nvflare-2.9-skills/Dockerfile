FROM nvflare-agent-eval:basic

RUN python -m pip install "git+https://github.com/chesterxgchen/NVFlare.git@flare_agent" \
    && nvflare agent skills install --agent codex \
    && nvflare agent skills install --agent claude \
    && nvflare agent skills install --agent codex --target /home/dev/.hermes/skills
