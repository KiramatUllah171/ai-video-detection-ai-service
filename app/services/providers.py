from __future__ import annotations

import mimetypes
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings, settings
from app.models.requests import AnalyzeFramesRequest, AnalyzeVideoRequest
from app.models.responses import AnalyzeFramesResponse
from app.services.base_ai_service import BaseAiService
from app.services.score_mapping import label_hint
from app.utils.errors import AiServiceError

logger = logging.getLogger(__name__)


class BitMindProviderError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw_response: dict | None = None,
        error_code: str = "BITMIND_UNAVAILABLE",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw_response = raw_response
        self.error_code = error_code


class BaseAiProvider:
    def analyze_video(self, request: AnalyzeVideoRequest) -> AnalyzeFramesResponse:
        raise NotImplementedError


class LocalAiProvider(BaseAiProvider):
    def __init__(self, service: BaseAiService) -> None:
        self.service = service

    def analyze_video(self, request: AnalyzeVideoRequest) -> AnalyzeFramesResponse:
        if not request.frames:
            raise AiServiceError(
                "FRAME_IMAGE_REQUIRED",
                "Local AI analysis requires extracted frame images.",
                status_code=400,
            )

        response = self.service.analyze_frames(
            AnalyzeFramesRequest(video_id=request.video_id, job_id=request.job_id, frames=request.frames)
        )
        return response.model_copy(update={
            "provider": "Local",
            "provider_mode": request.provider_mode,
            "final_decision_source": "Local",
        })


class BitMindAiProvider(BaseAiProvider):
    application_header = "oracle-api"

    def __init__(self, current_settings: Settings = settings) -> None:
        self.settings = current_settings

    def analyze_video(self, request: AnalyzeVideoRequest) -> AnalyzeFramesResponse:
        if not self.settings.bitmind_enabled:
            raise BitMindProviderError("BitMind provider is disabled.")
        if not self.settings.bitmind_api_key:
            raise BitMindProviderError("BitMind API key is not configured.")
        if not request.original_video_path and not request.video_url:
            raise BitMindProviderError("A local video path or video URL is required for BitMind analysis.")

        started_at = datetime.now(timezone.utc).isoformat()
        raw, transfer_metadata = self._detect(request)
        completed_at = datetime.now(timezone.utc).isoformat()
        is_ai = raw.get("isAI")
        confidence = _optional_score(raw.get("confidence"))
        normalized = _normalize_bitmind_score(is_ai, confidence)
        logger.info(
            "BitMind detection normalized for video_id=%s job_id=%s is_ai=%s confidence=%s final_label=%s decision_band=%s",
            request.video_id,
            request.job_id,
            is_ai,
            confidence,
            normalized["provider_label"],
            normalized["decision_band"],
        )
        external_result = {
            "provider_name": "BitMind",
            "provider_request_id": raw.get("request_id"),
            "provider_job_id": raw.get("job_id") or raw.get("id"),
            "provider_status": "Completed",
            "provider_label": normalized["provider_label"],
            "provider_score": normalized["provider_score"],
            "provider_confidence": confidence,
            "provider_decision_band": normalized["decision_band"],
            **transfer_metadata,
            "provider_raw_response": raw,
            "provider_started_at": started_at,
            "provider_completed_at": completed_at,
            "provider_error_message": None,
        }

        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id="bitmind-subnet-34",
            model_version="bitmind-oracle-v1-sn34",
            model_capability="external_video",
            is_mock=False,
            overall_ai_score=round(normalized["ai_score"], 4),
            real_probability=round(normalized["real_probability"], 4),
            overall_confidence=round(normalized["provider_confidence"], 4),
            label_hint=normalized["provider_label"],
            frames=[],
            notes=["External BitMind video detection completed.", normalized["summary"]],
            warnings=[
                "This video may be processed by an external AI detection provider for analysis.",
                *normalized["warnings"],
                *(["A compressed analysis copy was sent to BitMind because the original video exceeded the provider upload limit."]
                  if transfer_metadata.get("compression_used") else []),
            ],
            provider="BitMind",
            provider_mode=request.provider_mode,
            external_provider_result=external_result,
            bitmind_result=external_result,
            final_decision_source="BitMind",
        )

    def _detect(self, request: AnalyzeVideoRequest) -> dict[str, Any]:
        base_url = self.settings.bitmind_base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.settings.bitmind_api_key}",
            "x-bitmind-application": self.application_header,
        }
        timeout = httpx.Timeout(float(self.settings.bitmind_timeout_seconds))
        transfer_metadata: dict[str, Any] = {
            "original_file_size_bytes": None,
            "bitmind_file_size_bytes": None,
            "compression_used": False,
            "compression_attempts": 0,
            "compression_error": None,
            "analysis_copy_path": None,
            "provider_sent_file_name": None,
        }
        with httpx.Client(timeout=timeout) as client:
            if request.video_url:
                transfer_metadata["provider_sent_file_name"] = "video_url"
                logger.info(
                    "Sending BitMind video URL request for video_id=%s job_id=%s url_present=%s",
                    request.video_id,
                    request.job_id,
                    bool(request.video_url),
                )
                response = client.post(
                    f"{base_url}/34/detect-video",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"video": request.video_url, "rich": True},
                )
            else:
                send_path, transfer_metadata, cleanup_path = self._prepare_video_file(request)
                try:
                    file_size = send_path.stat().st_size
                    if file_size > self.settings.bitmind_direct_upload_limit_bytes:
                        logger.info(
                            "Sending BitMind presigned-upload video request for video_id=%s job_id=%s path=%s file_size=%s compression_used=%s",
                            request.video_id,
                            request.job_id,
                            send_path,
                            file_size,
                            transfer_metadata["compression_used"],
                        )
                        video_url = self._upload_large_video(client, base_url, headers, send_path)
                        response = client.post(
                            f"{base_url}/34/detect-video",
                            headers={**headers, "Content-Type": "application/json"},
                            json={"video": video_url, "rich": True},
                        )
                    else:
                        content_type = mimetypes.guess_type(send_path.name)[0] or "video/mp4"
                        logger.info(
                            "Sending BitMind direct video upload for video_id=%s job_id=%s path=%s file_size=%s content_type=%s compression_used=%s",
                            request.video_id,
                            request.job_id,
                            send_path,
                            file_size,
                            content_type,
                            transfer_metadata["compression_used"],
                        )
                        with send_path.open("rb") as file_handle:
                            response = client.post(
                                f"{base_url}/34/detect-video",
                                headers=headers,
                                files={"video": (send_path.name, file_handle, content_type)},
                                data={"rich": "true"},
                            )
                finally:
                    if cleanup_path is not None:
                        _delete_quietly(cleanup_path)

        raw = _response_json(response)
        if response.status_code >= 400:
            message = _safe_error_message(_error_message(raw) or f"BitMind returned HTTP {response.status_code}.", response.status_code)
            raise BitMindProviderError(message, response.status_code, raw, _bitmind_error_code(response.status_code))
        return raw, transfer_metadata

    def _prepare_video_file(self, request: AnalyzeVideoRequest) -> tuple[Path, dict[str, Any], Path | None]:
        path = Path(request.original_video_path or "")
        if not path.exists():
            raise BitMindProviderError("Local video file was not found.")
        self._validate_video_path(path)

        original_size = path.stat().st_size
        metadata: dict[str, Any] = {
            "original_file_size_bytes": original_size,
            "bitmind_file_size_bytes": original_size,
            "compression_used": False,
            "compression_attempts": 0,
            "compression_error": None,
            "analysis_copy_path": None,
            "provider_sent_file_name": path.name,
        }

        if original_size <= self.settings.bitmind_max_upload_bytes:
            return path, metadata, None

        output_path: Path | None = None
        try:
            output_path, attempts = self._compress_for_bitmind(path)
            compressed_size = output_path.stat().st_size
            metadata.update({
                "bitmind_file_size_bytes": compressed_size,
                "compression_used": True,
                "compression_attempts": attempts,
                "analysis_copy_path": str(output_path),
                "provider_sent_file_name": output_path.name,
            })
            return output_path, metadata, output_path
        except Exception as exception:
            if output_path is not None:
                _delete_quietly(output_path)
            metadata.update({
                "compression_used": True,
                "compression_error": str(exception),
            })
            raise BitMindProviderError(
                "Video exceeded BitMind upload limit and compression failed. Local analysis was used.",
                raw_response=metadata,
            ) from exception

    def _validate_video_path(self, path: Path) -> None:
        allowed_roots = self.settings.allowed_video_roots
        if not allowed_roots:
            if self.settings.is_production:
                raise BitMindProviderError("AI_ALLOWED_VIDEO_ROOTS must be configured before local video paths are accepted.")
            return

        resolved_path = path.resolve()
        if not any(resolved_path == root or resolved_path.is_relative_to(root) for root in allowed_roots):
            raise BitMindProviderError("Local video path is outside the allowed analysis work directories.")

    def _compress_for_bitmind(self, input_path: Path) -> tuple[Path, int]:
        duration = self._probe_duration_seconds(input_path)
        if duration <= 0:
            raise BitMindProviderError("Could not determine video duration for BitMind compression.")

        audio_bitrate_kbps = 96
        target_mb = self.settings.bitmind_compression_target_bytes / (1024 * 1024)
        initial_video_bitrate = ((target_mb * 8192) / duration) - audio_bitrate_kbps
        video_bitrate = int(max(500, min(2500, initial_video_bitrate)))
        output_path = Path(tempfile.gettempdir()) / f"bitmind_analysis_{input_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.mp4"

        attempts = 0
        last_error: str | None = None
        for retry_index in range(3):
            attempts = retry_index + 1
            adjusted_bitrate = max(500, int(video_bitrate * (0.75 ** retry_index)))
            command = [
                self.settings.ffmpeg_path,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                "scale='min(1280,iw)':-2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-b:v",
                f"{adjusted_bitrate}k",
                "-maxrate",
                f"{adjusted_bitrate}k",
                "-bufsize",
                f"{adjusted_bitrate * 2}k",
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_bitrate_kbps}k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            logger.info(
                "Compressing BitMind analysis copy input=%s output=%s attempt=%s video_bitrate_kbps=%s",
                input_path,
                output_path,
                attempts,
                adjusted_bitrate,
            )
            result = subprocess.run(command, capture_output=True, text=True, timeout=max(60, int(duration * 3)))
            if result.returncode != 0:
                last_error = result.stderr[-1000:] if result.stderr else "ffmpeg compression failed"
                continue
            if not output_path.exists():
                last_error = "ffmpeg did not create compressed output"
                continue
            if output_path.stat().st_size <= self.settings.bitmind_max_upload_bytes:
                return output_path, attempts
            last_error = "compressed output remained above BitMind upload limit"

        _delete_quietly(output_path)
        raise BitMindProviderError(last_error or "BitMind compression failed.")

    def _probe_duration_seconds(self, input_path: Path) -> float:
        command = [
            self.settings.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise BitMindProviderError(result.stderr[-1000:] if result.stderr else "ffprobe duration failed")
        try:
            return float(result.stdout.strip())
        except ValueError as exception:
            raise BitMindProviderError("ffprobe returned invalid duration") from exception

    def _upload_large_video(self, client: httpx.Client, base_url: str, headers: dict[str, str], path: Path) -> str:
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        upload_response = client.post(
            f"{base_url}/34/get-video-upload-url",
            headers={**headers, "Content-Type": "application/json"},
            json={"filename": path.name, "contentType": content_type},
        )
        upload_json = _response_json(upload_response)
        if upload_response.status_code >= 400:
            raise BitMindProviderError(_error_message(upload_json) or "Could not create BitMind upload URL.", upload_response.status_code, upload_json)
        upload_url = upload_json.get("url")
        fields = upload_json.get("fields") or {}
        video_url = upload_json.get("videoUrl")
        if not upload_url or not video_url:
            raise BitMindProviderError("BitMind upload URL response was incomplete.", upload_response.status_code, upload_json)
        with path.open("rb") as file_handle:
            s3_response = client.post(upload_url, data=fields, files={"file": (path.name, file_handle, content_type)})
        if s3_response.status_code >= 400:
            raise BitMindProviderError("BitMind presigned upload failed.", s3_response.status_code, {"status_code": s3_response.status_code})
        return str(video_url)


class ProviderOrchestrator(BaseAiProvider):
    def __init__(self, local_provider: LocalAiProvider, bitmind_provider: BitMindAiProvider, current_settings: Settings = settings) -> None:
        self.local_provider = local_provider
        self.bitmind_provider = bitmind_provider
        self.settings = current_settings

    def analyze_video(self, request: AnalyzeVideoRequest) -> AnalyzeFramesResponse:
        mode = (request.provider_mode or self.settings.ai_provider or "local").lower()
        logger.info(
            "Provider orchestrator selected mode=%s for video_id=%s job_id=%s local_fallback_enabled=%s",
            mode,
            request.video_id,
            request.job_id,
            self.settings.local_fallback_enabled,
        )
        if mode == "bitmind":
            logger.info("Using BitMind provider for video_id=%s job_id=%s", request.video_id, request.job_id)
            return self._bitmind_with_fallback(request, None, "bitmind")
        if mode == "hybrid":
            return self._hybrid(request)
        logger.info("Using local provider for video_id=%s job_id=%s", request.video_id, request.job_id)
        return self.local_provider.analyze_video(request.model_copy(update={"provider_mode": "local"}))

    def _hybrid(self, request: AnalyzeVideoRequest) -> AnalyzeFramesResponse:
        local_result = self.local_provider.analyze_video(request.model_copy(update={"provider_mode": "hybrid"}))
        if not _should_use_bitmind(local_result, self.settings):
            return local_result.model_copy(update={
                "provider": "Local",
                "provider_mode": "hybrid",
                "final_decision_source": "Local",
                "external_provider_result": {
                    "provider_name": "BitMind",
                    "provider_status": "Skipped",
                    "provider_error_message": "Policy skipped external verification because local result was high confidence.",
                },
                "local_result": local_result.model_dump(exclude={"local_result", "bitmind_result"}),
            })

        bitmind_result = self._bitmind_with_fallback(request, local_result, "hybrid")
        if bitmind_result.final_decision_source == "FallbackLocal":
            return bitmind_result
        return _combine_hybrid(local_result, bitmind_result)

    def _bitmind_with_fallback(
        self,
        request: AnalyzeVideoRequest,
        local_result: AnalyzeFramesResponse | None,
        mode: str,
    ) -> AnalyzeFramesResponse:
        try:
            return self.bitmind_provider.analyze_video(request.model_copy(update={"provider_mode": mode}))
        except (BitMindProviderError, httpx.HTTPError) as exception:
            message = exception.message if isinstance(exception, BitMindProviderError) else "BitMind request failed."
            error_code = exception.error_code if isinstance(exception, BitMindProviderError) else "BITMIND_UNAVAILABLE"
            error_metadata = exception.raw_response if isinstance(exception, BitMindProviderError) and isinstance(exception.raw_response, dict) else {}
            if not self.settings.local_fallback_enabled:
                raise AiServiceError(error_code, f"External video analysis failed: {_safe_error_message(message, getattr(exception, 'status_code', None))}", status_code=502)
            fallback = local_result or self.local_provider.analyze_video(request.model_copy(update={"provider_mode": mode}))
            warnings = [
                *fallback.warnings,
                message if "compression failed" in message.lower() else "External BitMind verification failed/unavailable. Local analysis was used.",
            ]
            return fallback.model_copy(update={
                "provider": "Local",
                "provider_mode": mode,
                "warnings": list(dict.fromkeys(warnings)),
                "fallback_used": True,
                "fallback_reason": message,
                "external_provider_result": {
                    "provider_name": "BitMind",
                    "provider_status": "Failed",
                    "provider_error_message": message,
                    **error_metadata,
                },
                "local_result": fallback.model_dump(exclude={"local_result", "bitmind_result"}),
                "final_decision_source": "FallbackLocal",
            })


def _combine_hybrid(local_result: AnalyzeFramesResponse, bitmind_result: AnalyzeFramesResponse) -> AnalyzeFramesResponse:
    disagreement = _labels_disagree(local_result.overall_ai_score, bitmind_result.overall_ai_score)
    warnings = [*local_result.warnings, *bitmind_result.warnings]
    if disagreement:
        warnings.append("Local and BitMind providers disagree; final result was reduced to avoid overclaiming.")
        if max(local_result.overall_confidence, bitmind_result.overall_confidence) >= 0.75:
            final_score = 0.58 if bitmind_result.overall_ai_score >= local_result.overall_ai_score else 0.50
        else:
            final_score = 0.50
    else:
        final_score = (local_result.overall_ai_score * 0.4) + (bitmind_result.overall_ai_score * 0.6)
    confidence = min(0.95, (local_result.overall_confidence + bitmind_result.overall_confidence) / 2 + (0.05 if not disagreement else -0.08))
    final_score = _clamp(final_score)
    return bitmind_result.model_copy(update={
        "provider": "Hybrid",
        "provider_mode": "hybrid",
        "overall_ai_score": round(final_score, 4),
        "real_probability": round(1.0 - final_score, 4),
        "overall_confidence": round(_clamp(confidence), 4),
        "label_hint": label_hint(final_score),
        "warnings": list(dict.fromkeys(warnings)),
        "local_result": local_result.model_dump(exclude={"local_result", "bitmind_result"}),
        "bitmind_result": bitmind_result.model_dump(exclude={"local_result", "bitmind_result"}),
        "final_decision_source": "Hybrid",
        "model_disagreement": disagreement,
        "notes": ["Hybrid local and BitMind analysis completed."],
    })


def _should_use_bitmind(local_result: AnalyzeFramesResponse, current_settings: Settings) -> bool:
    policy = current_settings.normalized_external_provider_policy
    if policy == "disabled":
        return False
    if policy == "always":
        return True
    low_confidence = local_result.overall_confidence < 0.70
    suspicious = local_result.overall_ai_score >= 0.55
    inconclusive = 0.35 <= local_result.overall_ai_score < 0.55
    return low_confidence or local_result.model_disagreement or (
        suspicious and current_settings.bitmind_use_on_suspicious
    ) or (
        inconclusive and current_settings.bitmind_use_on_inconclusive
    )


def _labels_disagree(local_score: float, external_score: float) -> bool:
    return (local_score < 0.35 and external_score >= 0.65) or (local_score >= 0.65 and external_score < 0.35)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {"value": payload}
    except ValueError:
        return {"body": response.text[:1000]}


def _error_message(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "")
    return str(payload.get("message") or "") or None


def _bitmind_error_code(status_code: int | None) -> str:
    return {
        401: "BITMIND_AUTH_FAILED",
        403: "BITMIND_FORBIDDEN",
        429: "BITMIND_RATE_LIMITED",
    }.get(status_code, "BITMIND_UNAVAILABLE")


def _safe_error_message(message: str, status_code: int | None = None) -> str:
    cleaned = " ".join(str(message or "").split())
    if not cleaned:
        return "provider returned no error details."
    lowered = cleaned.lower()
    if status_code == 401 or "unauthorized" in lowered or "authorization" in lowered or "api key" in lowered or "token" in lowered:
        return "provider authentication failed."
    if status_code == 403 or "forbidden" in lowered or "permission" in lowered:
        return "provider rejected this request."
    if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return "provider rate limit was reached."
    return cleaned[:300]


def _normalize_bitmind_score(is_ai: Any, confidence: float | None) -> dict[str, Any]:
    if confidence is None:
        return {
            "ai_score": 0.5,
            "real_probability": 0.5,
            "provider_score": None,
            "provider_confidence": 0.0,
            "provider_label": "Inconclusive",
            "decision_band": "missing_confidence",
            "summary": "BitMind did not provide a confidence score, so the result is inconclusive.",
            "warnings": ["BitMind confidence was unavailable; score was not treated as real or AI-generated."],
        }

    if is_ai is True:
        ai_score = confidence
        if confidence >= 0.85:
            provider_label = "LikelyAiGenerated"
            decision_band = "ai_high_confidence"
            summary = "BitMind detected AI-like signals with high confidence."
            warnings = []
        elif confidence >= 0.65:
            provider_label = "Suspicious"
            decision_band = "ai_moderate_confidence"
            summary = "BitMind detected AI-like signals, but confidence is moderate. Treat this as suspicious, not definitive."
            warnings = ["BitMind detected AI-like signals, but confidence is moderate. This is suspicious, not definitive."]
        else:
            provider_label = "Inconclusive"
            decision_band = "ai_low_confidence"
            summary = "BitMind detected AI-like signals with low confidence, so the result is inconclusive."
            warnings = ["BitMind confidence was low; result is inconclusive."]
    elif is_ai is False:
        ai_score = 1.0 - confidence
        if confidence >= 0.85:
            provider_label = "LikelyReal"
            decision_band = "real_high_confidence"
            summary = "BitMind found real/authentic signals with high confidence."
            warnings = []
        elif confidence >= 0.65:
            provider_label = "Inconclusive"
            decision_band = "real_moderate_confidence"
            summary = "BitMind found real/authentic signals, but confidence is moderate."
            warnings = ["BitMind found real/authentic signals, but confidence is moderate."]
        else:
            provider_label = "Inconclusive"
            decision_band = "real_low_confidence"
            summary = "BitMind confidence was low, so the result is inconclusive."
            warnings = ["BitMind confidence was low; result is inconclusive."]
    else:
        ai_score = 0.5
        provider_label = "Inconclusive"
        decision_band = "missing_is_ai"
        summary = "BitMind did not provide a clear AI/real decision, so the result is inconclusive."
        warnings = ["BitMind did not provide a clear AI/real decision."]

    ai_score = _clamp(ai_score)
    return {
        "ai_score": ai_score,
        "real_probability": _clamp(1.0 - ai_score),
        "provider_score": ai_score,
        "provider_confidence": confidence,
        "provider_label": provider_label,
        "decision_band": decision_band,
        "summary": summary,
        "warnings": warnings,
    }


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return None


def _delete_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.warning("Could not delete temporary BitMind analysis copy at %s", path)


def _clamp(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0
