"""Lightweight internationalization utilities for the Result GUI.

This module provides a simple translation layer with no external dependencies.
It exposes functions to query supported languages, look up translated strings,
and extract language-specific dictionaries for client-side use.

All text should use stable translation keys. English strings serve as a
fallback when a translation is missing.
"""
from __future__ import annotations

from typing import Dict, Iterable

SUPPORTED_LANGS: tuple[str, ...] = ("en", "ko")

# Central translation catalog. Keys are stable identifiers used in templates
# and scripts. Values are per-language strings.
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # App and navigation
    "app.title": {"en": "Result GUI", "ko": "결과 GUI"},
    "nav.results": {"en": "Results", "ko": "결과"},
    "nav.admin": {"en": "Admin", "ko": "관리자"},
    "nav.manage_users": {"en": "Manage Users", "ko": "사용자 관리"},
    "nav.login": {"en": "Login", "ko": "로그인"},
    "nav.logout": {"en": "Logout", "ko": "로그아웃"},
    "nav.profile": {"en": "Profile", "ko": "프로필"},
    "nav.lang.en": {"en": "English", "ko": "영어"},
    "nav.lang.ko": {"en": "Korean", "ko": "한국어"},
    "nav.theme": {"en": "Theme", "ko": "테마"},
    "theme.light": {"en": "Light", "ko": "라이트"},
    "theme.dark": {"en": "Dark", "ko": "다크"},
    "theme.auto": {"en": "Auto", "ko": "자동"},
    "theme.retro": {"en": "Retro Futurism", "ko": "레트로 퓨처리즘"},
    "theme.cyberpunk": {"en": "Cyberpunk", "ko": "사이버펑크"},
    "theme.glass": {"en": "Glassmorphism", "ko": "글래스모피즘"},

    # Common labels
    "common.entry_number": {"en": "Entry Number", "ko": "항목 번호"},
    "common.username": {"en": "Username", "ko": "사용자명"},
    "common.action": {"en": "Action", "ko": "작업"},
    "common.ip": {"en": "IP", "ko": "IP"},
    "common.user_agent": {"en": "User Agent", "ko": "사용자 에이전트"},
    "common.timestamp": {"en": "Timestamp", "ko": "타임스탬프"},
    "common.user_id": {"en": "User ID", "ko": "사용자 ID"},
    "common.total": {"en": "Total", "ko": "총계"},
    "common.used": {"en": "Used", "ko": "사용됨"},
    "common.free": {"en": "Free", "ko": "여유"},
    "common.status": {"en": "Status", "ko": "상태"},
    "common.error": {"en": "Error", "ko": "오류"},
    "common.connected": {"en": "Connected", "ko": "연결됨"},
    "common.backlog": {"en": "Backlog", "ko": "대기열"},
    "common.hostname": {"en": "Hostname", "ko": "호스트명"},
    "common.platform": {"en": "Platform", "ko": "플랫폼"},
    "common.uptime_s": {"en": "Uptime (s)", "ko": "가동 시간(초)"},
    "common.load": {"en": "Load (1m/5m/15m)", "ko": "부하 (1분/5분/15분)"},
    "common.last_updated": {"en": "Last updated:", "ko": "마지막 업데이트:"},
    "common.page": {"en": "Page", "ko": "페이지"},
    "common.of": {"en": "of", "ko": "/"},
    "common.yes": {"en": "Yes", "ko": "예"},
    "common.no": {"en": "No", "ko": "아니오"},
    "common.none": {"en": "(none)", "ko": "(없음)"},
    "common.default": {"en": "(default)", "ko": "(기본)"},
    "common.online": {"en": "online", "ko": "온라인"},
    "common.offline": {"en": "offline", "ko": "오프라인"},
    "common.direction": {"en": "Direction", "ko": "방향"},
    "common.asc": {"en": "Asc", "ko": "오름차순"},
    "common.desc": {"en": "Desc", "ko": "내림차순"},
    "common.apply": {"en": "Apply", "ko": "적용"},

    # Index page
    "index.title": {"en": "Results", "ko": "결과"},
    "index.global_search": {"en": "Global Search", "ko": "전체 검색"},
    "index.search_placeholder": {"en": "Search across all columns", "ko": "모든 열에서 검색"},
    "index.column": {"en": "Column", "ko": "열"},
    "index.mode": {"en": "Mode", "ko": "모드"},
    "index.mode.equals": {"en": "equals", "ko": "일치"},
    "index.mode.contains": {"en": "contains", "ko": "포함"},
    "index.value": {"en": "Value", "ko": "값"},
    "index.filter_results": {"en": "Filter Results", "ko": "결과 필터"},
    "index.field": {"en": "Field", "ko": "필드"},
    "index.field.top_similar_docs": {"en": "top_similar_docs", "ko": "top_similar_docs"},
    "index.field.image_authenticity": {"en": "image_authenticity", "ko": "image_authenticity"},
    "index.doc_key": {"en": "Doc Key", "ko": "문서 키"},
    "index.doc_key_placeholder": {"en": "document key (optional)", "ko": "문서 키 (선택)"},
    "index.key_mode": {"en": "Key Mode", "ko": "키 모드"},
    "index.score_op": {"en": "Score Op", "ko": "점수 연산"},
    "index.score": {"en": "Score", "ko": "점수"},
    "index.score_placeholder": {"en": "e.g., 0.9", "ko": "예: 0.9"},
    "index.class": {"en": "Class", "ko": "클래스"},
    "index.class_mode": {"en": "Class Mode", "ko": "클래스 모드"},
    "index.class_placeholder": {"en": "class (e.g., recaptured)", "ko": "클래스 (예: 재촬영)"},
    "index.sort_by": {"en": "Sort By", "ko": "정렬 기준"},
    "index.direction": {"en": "Direction", "ko": "정렬 방향"},
    "index.asc": {"en": "asc", "ko": "오름차순"},
    "index.desc": {"en": "desc", "ko": "내림차순"},
    "index.apply": {"en": "Apply", "ko": "적용"},
    "index.no_results": {"en": "No results", "ko": "결과 없음"},
    "index.previous": {"en": "Previous", "ko": "이전"},
    "index.next": {"en": "Next", "ko": "다음"},
    "index.none": {"en": "(none)", "ko": "(없음)"},
    "index.default": {"en": "(default)", "ko": "(기본)"},

    # Admin page
    "admin.total_users": {"en": "Total Users", "ko": "총 사용자"},
    "admin.signups_7d": {"en": "Sign-ups (7 days)", "ko": "가입 (7일)"},
    "admin.active_sessions_12h": {"en": "Active Sessions (12h)", "ko": "활성 세션 (12시간)"},
    "admin.recent_activity": {"en": "Recent Activity", "ko": "최근 활동"},
    "admin.total_active_sessions": {"en": "Total Active Sessions", "ko": "전체 활성 세션"},
    "admin.current_status": {"en": "Current Status", "ko": "현재 상태"},
    "admin.placeholder.status": {"en": "online or offline", "ko": "online 또는 offline"},
    "admin.add_user": {"en": "Add User", "ko": "사용자 추가"},
    "admin.system_health": {"en": "System Health", "ko": "시스템 상태"},
    "admin.export_system_health_csv": {"en": "Export System Health CSV", "ko": "시스템 상태 CSV 내보내기"},
    "admin.export_csv": {"en": "Export CSV", "ko": "CSV 내보내기"},
    "admin.refresh": {"en": "Refresh", "ko": "새로고침"},
    "admin.unable_fetch_metrics": {"en": "Unable to fetch system metrics.", "ko": "시스템 지표를 가져올 수 없습니다."},
    "admin.no_activity": {"en": "No activity yet", "ko": "활동 기록이 없습니다"},
    "admin.server": {"en": "Server", "ko": "서버"},
    "admin.database": {"en": "Database", "ko": "데이터베이스"},
    "admin.queue": {"en": "Queue", "ko": "큐"},
    "admin.storage": {"en": "Storage", "ko": "스토리지"},
    "admin.sort_by": {"en": "Sort by", "ko": "정렬 기준"},
    "admin.entry_number": {"en": "Entry", "ko": "항목"},
    "admin.unified_filter": {"en": "Filter", "ko": "필터"},
    "admin.additional_filters": {"en": "Additional filters", "ko": "추가 필터"},
    "admin.placeholder.username": {"en": "Enter username", "ko": "사용자명을 입력하세요"},
    "admin.placeholder.action": {"en": "Enter 'login' or 'logout'", "ko": "'login' 또는 'logout' 입력"},
    "admin.placeholder.ip": {"en": "Enter IP address", "ko": "IP 주소를 입력하세요"},
    "admin.placeholder.timestamp": {"en": "YYYY-MM-DD or YYYY-MM-DD HH:MM:SS", "ko": "YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS"},
    "admin.placeholder.user_agent": {"en": "Enter user agent substring", "ko": "User-Agent 일부를 입력하세요"},
    "admin.placeholder.user_id": {"en": "Enter user id (numeric)", "ko": "사용자 ID(숫자)를 입력하세요"},
    "admin.placeholder.entry": {"en": "Enter entry id (numeric)", "ko": "항목 ID(숫자)를 입력하세요"},

    # Admin - Clear Recent Activity
    "admin.clear_activity": {"en": "Clear Recent Activity", "ko": "최근 활동 지우기"},
    "admin.clear": {"en": "Clear", "ko": "지우기"},
    "admin.clear_all": {"en": "All", "ko": "전체"},
    "admin.clear_today": {"en": "Today", "ko": "오늘"},
    "admin.clear_past_days": {"en": "Past {days} day(s)", "ko": "지난 {days}일"},
    "admin.clear_confirm": {"en": "Delete selected activity logs?", "ko": "선택한 활동 로그를 삭제하시겠습니까?"},

    # Manage users
    "users.title": {"en": "Manage Users", "ko": "사용자 관리"},
    "users.search_placeholder": {"en": "Search username", "ko": "사용자명 검색"},
    "users.search": {"en": "Search", "ko": "검색"},
    "users.create_user": {"en": "Create User", "ko": "사용자 생성"},
    "users.username": {"en": "Username", "ko": "사용자명"},
    "users.password": {"en": "Password", "ko": "비밀번호"},
    "users.role": {"en": "Role", "ko": "역할"},
    "users.viewer": {"en": "viewer", "ko": "뷰어"},
    "users.admin": {"en": "admin", "ko": "관리자"},
    "users.create": {"en": "Create", "ko": "생성"},
    "users.id": {"en": "ID", "ko": "ID"},
    "users.created_at": {"en": "Created At", "ko": "생성 일시"},
    "users.actions": {"en": "Actions", "ko": "작업"},
    "users.update": {"en": "Update", "ko": "업데이트"},
    "users.set_password": {"en": "Set Password", "ko": "비밀번호 설정"},
    "users.delete": {"en": "Delete", "ko": "삭제"},
    "users.no_users": {"en": "No users", "ko": "사용자가 없습니다"},
    "users.delete_confirm": {"en": "Delete user {username}?", "ko": "사용자 {username}을(를) 삭제하시겠습니까?"},
    "users.new_password": {"en": "New password", "ko": "새 비밀번호"},

    # Login
    "login.title": {"en": "Login", "ko": "로그인"},
    "login.username": {"en": "Username", "ko": "사용자명"},
    "login.password": {"en": "Password", "ko": "비밀번호"},
    "login.sign_in": {"en": "Sign in", "ko": "로그인"},
    "login.remember_me": {"en": "Remember me", "ko": "로그인 상태 유지"},

    # Profile
    "profile.title": {"en": "Profile", "ko": "프로필"},
    "profile.change_password": {"en": "Change Password", "ko": "비밀번호 변경"},
    "profile.current_password": {"en": "Current Password", "ko": "현재 비밀번호"},
    "profile.new_password": {"en": "New Password", "ko": "새 비밀번호"},
    "profile.confirm_password": {"en": "Confirm Password", "ko": "비밀번호 확인"},
    "profile.update_password": {"en": "Update Password", "ko": "비밀번호 업데이트"},

    # Client-side controls
    "controls.hide_columns": {"en": "Hide columns", "ko": "열 숨기기"},
    "controls.rows_per_page": {"en": "Rows per page", "ko": "페이지당 행 수"},
    "controls.prev": {"en": "Prev", "ko": "이전"},
    "controls.next": {"en": "Next", "ko": "다음"},
    "controls.page": {"en": "Page", "ko": "페이지"},
    "controls.of": {"en": "of", "ko": "/"},
    "controls.simplify": {"en": "Simplify result", "ko": "결과 단순화"},
    "controls.export_csv": {"en": "Export CSV", "ko": "CSV 내보내기"},
    "controls.export_excel": {"en": "Export Excel", "ko": "Excel 내보내기"},
    "controls.hint_multi": {"en": "Hold Ctrl/Cmd to select multiple", "ko": "여러 항목을 선택하려면 Ctrl/Cmd 키를 누르세요"},

    # Accessibility and ARIA labels
    "a11y.pass_show": {"en": "Show password", "ko": "비밀번호 표시"},
    "a11y.pass_hide": {"en": "Hide password", "ko": "비밀번호 숨기기"},

    # Alerts / flash messages
    "alerts.invalid_credentials": {"en": "Invalid username or password", "ko": "아이디 또는 비밀번호가 올바르지 않습니다"},
    "alerts.username_password_required": {"en": "Username and password are required", "ko": "사용자명과 비밀번호가 필요합니다"},
    "alerts.invalid_role": {"en": "Invalid role", "ko": "잘못된 역할입니다"},
    "alerts.role_updated": {"en": "Role updated", "ko": "역할이 업데이트되었습니다"},
    "alerts.user_created": {"en": "User '{username}' created", "ko": "사용자 '{username}'가 생성되었습니다"},
    "alerts.password_updated": {"en": "Password updated", "ko": "비밀번호가 업데이트되었습니다"},
    "alerts.user_deleted": {"en": "User deleted", "ko": "사용자가 삭제되었습니다"},
    "alerts.password_required": {"en": "Password is required", "ko": "비밀번호가 필요합니다"},
    "alerts.password_mismatch": {"en": "Passwords do not match", "ko": "비밀번호가 일치하지 않습니다"},
    "alerts.current_password_incorrect": {"en": "Current password is incorrect", "ko": "현재 비밀번호가 올바르지 않습니다"},
    "alerts.activity_cleared": {"en": "Deleted {count} activity record(s)", "ko": "{count}개의 활동 기록을 삭제했습니다"},
    "alerts.activity_clear_invalid": {"en": "Invalid selection; no records deleted", "ko": "잘못된 선택입니다. 삭제된 기록이 없습니다"},

    # 2FA (Two-Factor Authentication)
    "2fa.title": {"en": "Two-Factor Authentication", "ko": "2단계 인증"},
    "2fa.setup.instructions": {"en": "Scan the QR code with your authenticator app or enter the secret manually, then enter the 6-digit code to confirm.", "ko": "인증 앱으로 QR 코드를 스캔하거나 시크릿을 직접 입력한 후, 6자리 코드를 입력하세요."},
    "2fa.verify.instructions": {"en": "Enter the 6-digit code from your authenticator app to continue.", "ko": "인증 앱에서 생성된 6자리 코드를 입력하세요."},
    "2fa.secret": {"en": "Secret", "ko": "시크릿"},
    "2fa.code": {"en": "Authentication Code", "ko": "인증 코드"},
    "2fa.verify": {"en": "Verify", "ko": "확인"},
    "2fa.code.invalid": {"en": "Invalid authentication code. Please try again.", "ko": "잘못된 인증 코드입니다. 다시 시도하세요."},
    "2fa.setup.unavailable": {"en": "Two-factor setup is not available at this time.", "ko": "현재 2단계 인증 설정을 사용할 수 없습니다."},
    "2fa.enabled.hint": {"en": "After password, you will be asked for an authenticator code.", "ko": "비밀번호 입력 후 인증 앱의 코드를 요구합니다."},
}


def normalize_lang(code: str | None) -> str:
    """Normalize a language code to a supported value.

    Args:
        code: Incoming language code (may be None).

    Returns:
        A supported language code; defaults to "en" when unknown.
    """
    if not code:
        return "en"
    lc = code.lower()
    if lc in SUPPORTED_LANGS:
        return lc
    if lc.startswith("ko"):
        return "ko"
    return "en"


def t(key: str, lang: str) -> str:
    """Translate a key to the given language with English fallback.

    Args:
        key: Translation key.
        lang: Target language code (e.g., "en" or "ko").

    Returns:
        The translated string, or the key itself if not found.
    """
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry.get("en") or key


def keys() -> Iterable[str]:
    """Return all translation keys.

    Returns:
        Iterable of keys present in the catalog.
    """
    return TRANSLATIONS.keys()


def translations_for(lang: str) -> Dict[str, str]:
    """Return a flattened mapping for a language for client-side use.

    Args:
        lang: Target language code.

    Returns:
        A dictionary mapping keys to translated strings in the given language,
        with English fallback applied.
    """
    out: Dict[str, str] = {}
    for k, per_lang in TRANSLATIONS.items():
        out[k] = per_lang.get(lang) or per_lang.get("en") or k
    return out
