from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from reasoning_motifs_web.fixtures import DEFAULT_WEBAPP_ARTIFACT_DIR, REPO_ROOT
from reasoning_motifs_web.models import CorpusOverview, QuestionDetail, QuestionSummary


class ArtifactStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.question_dir = data_dir / "question"
        self.overview_path = data_dir / "overview.json"
        self.questions_path = data_dir / "questions.json"
        self._validate_layout()
        self.overview = CorpusOverview.model_validate_json(
            self.overview_path.read_text(encoding="utf-8")
        )
        question_rows = json.loads(self.questions_path.read_text(encoding="utf-8"))
        self.questions = [QuestionSummary.model_validate(row) for row in question_rows]

    def _validate_layout(self) -> None:
        missing = [
            str(path.name)
            for path in (self.overview_path, self.questions_path)
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                f"Artifact directory {self.data_dir} is missing required files: {', '.join(missing)}"
            )
        if not self.question_dir.exists():
            raise RuntimeError(
                f"Artifact directory {self.data_dir} is missing required question directory."
            )

    def load_question(self, question_id: str) -> QuestionDetail:
        path = self.question_dir / f"{question_id}.json"
        if not path.exists():
            raise FileNotFoundError(question_id)
        return QuestionDetail.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env_value = os.getenv("REASONING_MOTIFS_WEBAPP_DATA_DIR")
    if env_value:
        return Path(env_value)
    if DEFAULT_WEBAPP_ARTIFACT_DIR.exists():
        return DEFAULT_WEBAPP_ARTIFACT_DIR
    fallback_dir = REPO_ROOT / "webapp_artifacts" / "pilot_v1"
    return fallback_dir


def create_app(data_dir: Path | None = None) -> FastAPI:
    resolved_dir = _resolve_data_dir(data_dir)
    store = ArtifactStore(resolved_dir)

    app = FastAPI(
        title="Reasoning Motifs Webapp API",
        version="0.1.0",
    )
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/overview", response_model=CorpusOverview)
    def get_overview() -> CorpusOverview:
        return app.state.store.overview

    @app.get("/api/questions", response_model=list[QuestionSummary])
    def get_questions() -> list[QuestionSummary]:
        return app.state.store.questions

    @app.get("/api/questions/{question_id}", response_model=QuestionDetail)
    def get_question(question_id: str) -> QuestionDetail:
        try:
            return app.state.store.load_question(question_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown question_id: {exc.args[0]}") from exc

    return app


app = create_app()
