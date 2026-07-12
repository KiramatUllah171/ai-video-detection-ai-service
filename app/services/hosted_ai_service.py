from app.models.requests import AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse
from app.services.base_ai_service import BaseAiService
from app.utils.errors import AiServiceError


class HostedAiService(BaseAiService):
    model_capability = "hosted"

    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        raise AiServiceError(
            "HOSTED_INFERENCE_FAILED",
            "Hosted inference is not enabled in this development build. Configure local real mode or mock mode.",
            status_code=503,
        )
