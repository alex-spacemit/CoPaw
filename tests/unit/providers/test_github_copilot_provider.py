# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

from copaw.providers.github_copilot_provider import GitHubCopilotProvider


def _make_provider() -> GitHubCopilotProvider:
    return GitHubCopilotProvider(
        id="github-copilot",
        name="GitHub Copilot",
        base_url="https://api.githubcopilot.com",
        require_api_key=False,
        support_model_discovery=True,
        freeze_url=True,
    )


async def test_start_device_authorization_stores_session(monkeypatch) -> None:
    provider = _make_provider()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "device_code": "device-code-1",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def post(self, *args, **kwargs):
            _ = args, kwargs
            return FakeResponse()

    monkeypatch.setattr(
        "copaw.providers.github_copilot_provider.httpx.AsyncClient",
        FakeAsyncClient,
    )

    session = await provider.start_device_authorization()

    assert session.user_code == "ABCD-EFGH"
    assert session.interval == 5
    assert provider._device_sessions[session.session_id].device_code == "device-code-1"


async def test_poll_device_authorization_authorized(monkeypatch) -> None:
    provider = _make_provider()
    session = SimpleNamespace(
        session_id="session-1",
        device_code="device-code-1",
        user_code="ABCD-EFGH",
        verification_uri="https://github.com/login/device",
        expires_at=4_102_444_800,
        interval=5,
        status="pending",
        last_message="",
    )
    provider._device_sessions[session.session_id] = session

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "access_token": "gho_test_token",
                "token_type": "bearer",
                "scope": "read:user",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def post(self, *args, **kwargs):
            _ = args, kwargs
            return FakeResponse()

    async def fake_populate_user(timeout: float = 10) -> None:
        _ = timeout
        provider.github_user_login = "octocat"

    async def fake_refresh_copilot(timeout: float = 10) -> None:
        _ = timeout
        provider.copilot_access_token = "copilot_token"
        provider.copilot_token_expires_at = 4_102_444_800
        provider.api_key = "copilot_token"

    monkeypatch.setattr(
        "copaw.providers.github_copilot_provider.httpx.AsyncClient",
        FakeAsyncClient,
    )
    monkeypatch.setattr(provider, "_populate_github_user", fake_populate_user)
    monkeypatch.setattr(
        provider,
        "_refresh_copilot_token_async",
        fake_refresh_copilot,
    )

    status, message = await provider.poll_device_authorization("session-1")

    assert status == "authorized"
    assert message == "GitHub authorization completed"
    assert provider.github_oauth_token == "gho_test_token"
    assert provider.github_user_login == "octocat"
    assert provider.api_key == "copilot_token"
    assert "session-1" not in provider._device_sessions


async def test_get_info_reports_auth_state() -> None:
    provider = _make_provider()
    provider.github_oauth_token = "gho_test"
    provider.github_user_login = "octocat"
    provider.copilot_access_token = "copilot-token"
    provider.copilot_token_expires_at = 4_102_444_800
    provider.api_key = "copilot-token"

    info = await provider.get_info(mock_secret=False)

    assert info.supports_oauth_login is True
    assert info.is_authenticated is True
    assert info.auth_account_label == "octocat"
    assert info.auth_expires_at == 4_102_444_800
    assert info.api_key == ""


def test_logout_clears_auth_state() -> None:
    provider = _make_provider()
    provider.github_oauth_token = "gho_test"
    provider.github_user_login = "octocat"
    provider.copilot_access_token = "copilot-token"
    provider.copilot_token_expires_at = 4_102_444_800
    provider.api_key = "copilot-token"
    provider._device_sessions["session-1"] = SimpleNamespace()

    provider.logout()

    assert provider.github_oauth_token == ""
    assert provider.github_user_login == ""
    assert provider.copilot_access_token == ""
    assert provider.copilot_token_expires_at is None
    assert provider.api_key == ""
    assert provider._device_sessions == {}


async def test_fetch_models_requires_and_uses_auth(monkeypatch) -> None:
    provider = _make_provider()
    provider.github_oauth_token = "gho_test"

    async def fake_refresh(timeout: float = 5) -> None:
        _ = timeout
        provider.copilot_access_token = "copilot-token"
        provider.api_key = "copilot-token"

    class FakeModels:
        async def list(self, timeout=None):
            _ = timeout
            return SimpleNamespace(
                data=[SimpleNamespace(id="gpt-4o", name="GPT-4o")],
            )

    monkeypatch.setattr(provider, "_refresh_copilot_token_async", fake_refresh)
    monkeypatch.setattr(
        provider,
        "_client",
        lambda timeout=5: SimpleNamespace(models=FakeModels()),
    )

    models = await provider.fetch_models(timeout=3)

    assert [model.id for model in models] == ["gpt-4o"]