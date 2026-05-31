# Multi-stage build for the heuristiX research platform.
#   stage 1 — compile the Rust DES simulator
#   stage 2 — Python runtime with FastAPI + Streamlit + the prebuilt binary

# ---- 1) Rust build ------------------------------------------------------
FROM rust:1.83-slim AS rust-build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY data ./data
RUN cargo build --release --bin heuristix


# ---- 2) Python runtime --------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HX_BINARY=/app/target/release/heuristix

WORKDIR /app

# Python deps
COPY hub/requirements.txt /tmp/req-hub.txt
COPY evolution/requirements.txt /tmp/req-evo.txt 2>/dev/null || true
RUN pip install --no-cache-dir -r /tmp/req-hub.txt \
    && pip install --no-cache-dir openai pyparsing

# Source — both the simulator wrapper Python and the platform code.
COPY evolution ./evolution
COPY hub ./hub
COPY runs/presentation ./runs/presentation 2>/dev/null || true

# Compiled Rust binary
COPY --from=rust-build /src/target/release/heuristix /app/target/release/heuristix

# Workspace dir (mountable as volume in docker-compose)
RUN mkdir -p /app/hub/workspace /app/runs

EXPOSE 8000 8501

# Default command runs both API and UI via a tiny entrypoint script.
COPY hub/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
