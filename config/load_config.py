# src/config/loader.py
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from config.auth_git import prepare_git_auth
from util.util_basic import (
    require_dict,
    validate_optional_bool,
    validate_optional_str,
    validate_required_str,
)

# ------ Data ------


@dataclass(frozen=True)
class RepoConfig:
    url: str
    target_branch: str
    token: str  # env에서 해석된 실제 토큰
    source_path: Optional[str] = None  # internal
    target_path: Optional[str] = None  # external


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path = Path("./out")


@dataclass(frozen=True)
class FinalConfig:
    internal: RepoConfig
    external: RepoConfig
    file_list: List[str]
    logging: LoggingConfig
    report: ReportConfig
    build_file_list: bool = False


# ------ DEFAULT_CONFIG ------
DEFAULT_CONFIG: Dict[str, Any] = {
    "logging": {"enabled": True},
    "report": {"output_dir": "./out"},
    "build_file_list": False,
}


# ------ errors ------
class ConfigError(Exception):
    pass


class ConfigParseError(ConfigError):
    pass


class ConfigValidationError(ConfigError):
    pass


class EnvVarMissingError(ConfigError):
    pass


# ------ config core ------
def load_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ConfigParseError("config.json 최상위는 JSON object여야 합니다.")
        return data
    except json.JSONDecodeError as e:
        raise ConfigParseError(f"config.json 문법 오류: {e}") from e
    except OSError as e:
        raise ConfigParseError(f"config.json 읽기 실패: {e}") from e


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """base에 override를 덮어쓰되, dict는 재귀적으로 병합(사용자 값 우선)."""
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(DEFAULT_CONFIG, raw)


def resolve_token(token_env: str) -> str:
    val = os.environ.get(token_env)
    if not val:
        raise EnvVarMissingError(f"환경변수 '{token_env}'가 설정되지 않았습니다.")
    return val


def validate_config(cfg: Dict[str, Any]) -> None:
    # 필수 항목
    validate_required_str(cfg, "repos.internal.url", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.external.url", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.internal.target_branch", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.external.target_branch", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.internal.source_path", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.external.target_path", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.internal.token_env", exc=ConfigValidationError)
    validate_required_str(cfg, "repos.external.token_env", exc=ConfigValidationError)

    # file_list
    file_list = cfg.get("file_list")
    if not isinstance(file_list, list) or len(file_list) == 0:
        raise ConfigValidationError("file_list는 빈 리스트가 될 수 없습니다(최소 1개 이상).")
    if any((not isinstance(x, str) or not x.strip()) for x in file_list):
        raise ConfigValidationError("file_list 항목은 비어있지 않은 문자열이어야 합니다.")

    # ---- 선택 항목 타입 검증 (bool() 캐스팅 제거 목적) ----
    validate_optional_bool(cfg, "logging.enabled", default=True, exc=ConfigValidationError)
    validate_optional_bool(cfg, "build_file_list", default=False, exc=ConfigValidationError)
    validate_optional_str(cfg, "report.output_dir", default="./out", exc=ConfigValidationError)

    # repos 타입 보장 (normalize_paths에서 안전하게 접근하기 위함)
    require_dict(cfg, "repos", exc=ConfigValidationError)
    require_dict(cfg, "repos.internal", exc=ConfigValidationError)
    require_dict(cfg, "repos.external", exc=ConfigValidationError)


def normalize_paths(cfg: Dict[str, Any], base_dir: Path) -> None:
    # ---- 타입/키 방어: dict 보장 ----
    internal = require_dict(cfg, "repos.internal", exc=ConfigValidationError)
    external = require_dict(cfg, "repos.external", exc=ConfigValidationError)

    report = cfg.get("report")
    if report is None:
        cfg["report"] = {}
        report = cfg["report"]
    if not isinstance(report, dict):
        raise ConfigValidationError("타입 오류(dict 필요): report")

    # report.output_dir 정규화
    out_dir = report.get("output_dir", "./out")
    if not isinstance(out_dir, str):
        raise ConfigValidationError("타입 오류(str 필요): report.output_dir")

    out_path = (base_dir / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
    report["output_dir"] = str(out_path)

    # source_path / target_path 표현 통일
    internal["source_path"] = str(Path(internal["source_path"]))
    external["target_path"] = str(Path(external["target_path"]))


def build_final_config(cfg: Dict[str, Any]) -> FinalConfig:
    internal_token = resolve_token(cfg["repos"]["internal"]["token_env"])
    external_token = resolve_token(cfg["repos"]["external"]["token_env"])

    internal = RepoConfig(
        url=cfg["repos"]["internal"]["url"],
        target_branch=cfg["repos"]["internal"]["target_branch"],
        source_path=cfg["repos"]["internal"]["source_path"],
        token=internal_token,
    )
    external = RepoConfig(
        url=cfg["repos"]["external"]["url"],
        target_branch=cfg["repos"]["external"]["target_branch"],
        target_path=cfg["repos"]["external"]["target_path"],
        token=external_token,
    )

    # validate_config에서 타입을 보장했으므로 캐스팅 불필요
    logging_enabled = validate_optional_bool(
        cfg, "logging.enabled", default=True, exc=ConfigValidationError
    )
    build_file_list = validate_optional_bool(
        cfg, "build_file_list", default=False, exc=ConfigValidationError
    )

    logging_cfg = LoggingConfig(enabled=logging_enabled)

    # normalize_paths에서 str(절대경로 문자열)로 정규화했으므로 Path 변환만
    report_output_dir = validate_optional_str(
        cfg, "report.output_dir", default="./out", exc=ConfigValidationError
    )
    report_cfg = ReportConfig(output_dir=Path(report_output_dir))

    return FinalConfig(
        internal=internal,
        external=external,
        file_list=[x.strip() for x in cfg["file_list"]],
        logging=logging_cfg,
        report=report_cfg,
        build_file_list=build_file_list,
    )


def load_config(config_path: str) -> FinalConfig:
    raw = load_json_file(config_path)  # 1) 읽기
    merged = apply_defaults(raw)  # 2) 기본값 병합
    validate_config(merged)  # 4) 검증
    base_dir = Path(config_path).parent
    normalize_paths(merged, base_dir=base_dir)  # 5) 정규화(최종 반환 직전 포함)
    return build_final_config(merged)  # env 해석 + 최종 반환
