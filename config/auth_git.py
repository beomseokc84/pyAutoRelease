# src/config/auth_git.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True)
class GitAuth:
    """Git 인증 준비 결과.

    - auth_url: git 명령에서 사용할 인증 포함 URL (token 포함)
    - safe_url: 로그 출력용(토큰 마스킹)
    - env: git 실행 시 사용할 환경 변수(프롬프트 방지 등)
    """

    original_url: str
    auth_url: str
    safe_url: str
    env: Dict[str, str]
    provider: str


def detect_provider(repo_url: str) -> str:
    """URL 기반으로 Git provider를 추정합니다."""
    u = repo_url.lower()
    if "gitlab" in u:
        return "gitlab"
    if "github" in u:
        return "github"
    return "generic"


def _pick_username(provider: str) -> str:
    # GitLab: oauth2 / GitHub: x-access-token 이 실무에서 자주 사용됨
    if provider == "gitlab":
        return "oauth2"
    if provider == "github":
        return "x-access-token"
    return "oauth2"


def redact_token(text: str, token: str) -> str:
    if not token:
        return text
    return text.replace(token, "***REDACTED***")


def build_auth_url_https(repo_url: str, token: str, provider: Optional[str] = None) -> Tuple[str, str, str]:
    """HTTPS URL에 토큰을 삽입합니다.

    반환값: (auth_url, safe_url, provider)
    - repo_url이 SSH(git@, ssh://) 형태면 예외를 발생시킵니다.
    """
    if repo_url.startswith("git@") or repo_url.startswith("ssh://"):
        raise ValueError("SSH URL은 토큰 기반 인증 URL로 변환할 수 없습니다. HTTPS URL을 사용하세요.")

    parsed = urlparse(repo_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"지원하지 않는 URL scheme 입니다: {parsed.scheme}")

    provider_final = provider or detect_provider(repo_url)
    username = _pick_username(provider_final)

    # netloc에 username:token@host 형태로 삽입
    netloc = parsed.netloc
    if "@" in netloc:
        # 이미 userinfo가 있으면 제거 후 재구성
        netloc = netloc.split("@", 1)[1]

    auth_netloc = f"{username}:{token}@{netloc}"
    auth_url = urlunparse(parsed._replace(netloc=auth_netloc))

    safe_url = redact_token(auth_url, token)
    return auth_url, safe_url, provider_final


def build_git_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """git 실행 시 프롬프트/대화형 입력을 방지하는 환경 변수를 준비합니다."""
    env = dict(base_env or {})
    # 토큰이 없거나 권한 문제가 있을 때 git이 프롬프트를 띄우지 않도록 방지
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def prepare_git_auth(repo_url: str, token: str, provider: Optional[str] = None) -> GitAuth:
    """auth_git.py의 핵심: 토큰을 이용해 '인증 준비'만 수행합니다.
    실제 git clone/fetch/pull 실행은 sync_repo.py에서 담당합니다.
    """
    auth_url, safe_url, provider_final = build_auth_url_https(repo_url, token, provider=provider)
    env = build_git_env()
    return GitAuth(
        original_url=repo_url,
        auth_url=auth_url,
        safe_url=safe_url,
        env=env,
        provider=provider_final,
    )
