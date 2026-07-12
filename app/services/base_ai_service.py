from abc import ABC, abstractmethod

from app.models.requests import AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse


class BaseAiService(ABC):
    model_loaded: bool = False
    model_capability: str = "mock"

    @abstractmethod
    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        raise NotImplementedError
