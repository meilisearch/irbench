from pydantic import BaseModel, ConfigDict

from irbench.paths import slug

STAGES = ("prepare", "embed", "up", "index", "render", "relevance", "score", "down")


class RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: str
    engine: str
    model: str
    track: str = "default"
    hardware: str = "local"
    host: str = "localhost"

    @property
    def run_id(self) -> str:
        return "__".join(
            [slug(self.scenario), self.engine, slug(self.model), self.track, self.hardware]
        )
