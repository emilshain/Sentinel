"""
Response models for the Sentinel backend.

One structural rule runs through this file: every response that carries pipeline
output inherits from `DataSourceTagged`, which declares `data_source` as a
required field with no default. Omitting the honest live/cached tag is therefore
a validation error at serialization time, not a convention someone can forget.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The only two values demo_runner._stamp_data_source ever writes. Kept strict on
# purpose: if the pipeline ever emits a third tag, this must fail loudly rather
# than pass an unrecognised provenance claim through to a client.
DataSource = Literal["live_run", "cached_golden_run"]

JobStatus = Literal["pending", "running", "done", "failed"]


class DataSourceTagged(BaseModel):
    """Base for any response carrying pipeline output. `data_source` is required."""

    data_source: DataSource
    fallback_reason: Optional[str] = None


class TabResult(BaseModel):
    """
    demo_view.tab_result - Tab 1 of the dashboard.

    Declared fields document the contract in frontend/README.md; extra="allow"
    means anything the pipeline adds later still passes through untouched
    instead of being silently dropped.
    """

    model_config = ConfigDict(extra="allow")

    data_source: DataSource
    fallback_reason: Optional[str] = None
    verdict: Optional[str] = None
    risk_score: Optional[float] = None
    confirmed_trigger: Optional[str] = None
    confidence: Optional[float] = None
    trigger_class: Optional[Any] = None
    supporting_samples: Optional[List[Any]] = None
    detector_votes: Optional[Dict[str, Any]] = None
    votes_backdoored: Optional[int] = None
    votes_total: Optional[int] = None
    dataset_scope: Optional[str] = None
    dataset_samples: Optional[int] = None
    runtime_seconds: Optional[float] = None
    hypothesis_generator: Optional[str] = None
    hypothesis_is_mock: Optional[bool] = None


class DemoView(BaseModel):
    """The compact UI contract projected out of the full report."""

    model_config = ConfigDict(extra="allow")

    tab_result: TabResult
    tab_how_we_found_it: Dict[str, Any] = Field(default_factory=dict)


class ScanPayload(DataSourceTagged):
    """What /scan produces: demo_view plus the top-level honesty tag."""

    demo_view: DemoView

    @model_validator(mode="after")
    def _tags_must_agree(self) -> "ScanPayload":
        nested = self.demo_view.tab_result.data_source
        if nested != self.data_source:
            raise ValueError(
                f"data_source mismatch: top-level={self.data_source!r} but "
                f"demo_view.tab_result={nested!r}. The UI reads the nested field, "
                "so serving this would misreport a replay as a live run."
            )
        return self

    @classmethod
    def from_report(cls, report: Dict[str, Any]) -> "ScanPayload":
        """Project a full run_demo() report down to the scan response."""
        if "data_source" not in report:
            raise ValueError(
                "run_demo() returned a report with no 'data_source'. Refusing to "
                "serve an untagged result."
            )
        return cls(
            data_source=report["data_source"],
            fallback_reason=report.get("fallback_reason"),
            demo_view=report.get("demo_view"),
        )


class JobSubmission(BaseModel):
    """202 response from POST /scan."""

    job_id: str
    status: JobStatus
    poll_url: str


class JobState(BaseModel):
    """
    GET /scan/{job_id}. `result` carries the exact payload /scan would have
    returned synchronously, and is present exactly when status == "done".
    """

    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    result: Optional[ScanPayload] = None

    @model_validator(mode="after")
    def _done_implies_result(self) -> "JobState":
        if self.status == "done" and self.result is None:
            raise ValueError("job marked done but carries no result payload")
        if self.status == "failed" and not self.error:
            raise ValueError("job marked failed but carries no error message")
        return self


class ReportResponse(DataSourceTagged):
    """
    GET /report - the full report dict (~180 KB of raw evidence).

    The report is nested under `report` rather than returned bare so the
    data_source guarantee is enforced by the same base model as everywhere else.
    `origin` says where this copy came from, since the on-disk file outlives the
    process and may predate it.
    """

    origin: Literal["in_memory_last_scan", "reports/demo_run_output.json"]
    report: Dict[str, Any]

    @model_validator(mode="after")
    def _tag_matches_report(self) -> "ReportResponse":
        inner = self.report.get("data_source")
        if inner != self.data_source:
            raise ValueError(
                f"data_source mismatch: envelope={self.data_source!r} but "
                f"report={inner!r}."
            )
        return self

    @classmethod
    def from_report(cls, report: Dict[str, Any], origin: str) -> "ReportResponse":
        if "data_source" not in report:
            raise ValueError(
                "Report has no 'data_source'. Refusing to serve an untagged result."
            )
        return cls(
            data_source=report["data_source"],
            fallback_reason=report.get("fallback_reason"),
            origin=origin,
            report=report,
        )


class Check(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    # `model_checkpoint_present` collides with pydantic's protected "model_"
    # namespace; the field name is worth keeping, the warning is not.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    cwd: str
    cwd_ok: bool
    model_checkpoint_present: bool
    api_key_set: bool
    golden_run_present: bool
    live_run_possible: bool
    can_serve_scan: bool
    scan_in_progress: bool
    checks: Dict[str, Check]
