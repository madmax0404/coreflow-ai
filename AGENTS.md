# Repository Guidelines

## Project Structure & Module Organization
Core runtime lives in `src/`, with `openai_agent_api.py` exposing the FastAPI surface that orchestrates `Agent` and `Runner`. MCP tooling and weather/RAG utilities stay under `src/agents/mcp/`; treat `server.py` as the authoritative entry point and `test_server.py` as the smoke harness for new tools. Put retrieval corpora in `data/`, notebooks in `notebooks/`, assets in `images/`, and keep scratch work in `tmp_*.py` so production imports stay clean.

## Build, Test, and Development Commands
Install dependencies once with `uv sync` (Python 3.13 is required). Run the HTTP agent bridge using `uv run uvicorn src.openai_agent_api:app --reload` and start the MCP tool host via `uv run python src/agents/mcp/server.py stdio`. For quick end-to-end checks, run `uv run python tmp_rag.py` after loading a valid `.env` so embeddings and weather endpoints respond.

## Coding Style & Naming Conventions
Match PEP 8/PEP 484: four-space indentation, snake_case for modules, PascalCase for Pydantic models, and UPPER_SNAKE_CASE for environment keys. Keep FastAPI payloads typed and validated, and prefer small helpers over long procedural blocks inside endpoints. Leave inline comments only when the flow is non-obvious (retry loops, external service quirks).

## Testing Guidelines
Extend `src/agents/mcp/test_server.py` or add files under `tests/` named `test_<feature>.py`. Cover tool registration, external API fallbacks, and RAG retrieval paths. Run suites with `uv run pytest` (add pytest to dev dependencies if missing) and document any manual validation in the PR until coverage improves.

## Commit & Pull Request Guidelines
Mirror the history that prefixes commits with the working date (e.g., `20250919 작업 내용`) plus a concise summary. PRs should explain the scenario, list new commands or environment variables, and link tracking issues; attach screenshots or console transcripts when agent behavior changes. Before review, confirm `.env` secrets stay local, note required external services, and report test or smoke results.

## Security & Configuration Tips
Load secrets through `.env` (see keys referenced in `server.py`) and never commit real API tokens. Document new external endpoints or ports in the PR description and prefer environment variables over hard-coded hosts. Share redacted example configs, using safe defaults such as `http://localhost:8000`.
