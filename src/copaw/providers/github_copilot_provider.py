# -*- coding: utf-8 -*-
"""GitHub Copilot provider with GitHub device authorization."""

from __future__ import annotations

import secrets
import time
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from .openai_chat_model_compat import OpenAIChatModelCompat
from .openai_provider import OpenAIProvider
from .provider import ModelInfo, ProviderInfo


GITHUB_COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
GITHUB_COPILOT_API_URL = "https://api.githubcopilot.com"
DEFAULT_COPILOT_TOKEN_TTL = 25 * 60
COPILOT_TOKEN_REFRESH_SKEW = 60


class DeviceAuthorizationSession(BaseModel):
    session_id: str
    device_code: str
    user_code: str
    verification_uri: str
    expires_at: int
    interval: int = 5
    status: str = "pending"
    last_message: str = ""


class GitHubCopilotProvider(OpenAIProvider):
    """OpenAI-compatible provider backed by GitHub Copilot."""

    supports_oauth_login: bool = True
    auth_method: str | None = "github-device"
    is_authenticated: bool = False
    auth_account_label: str | None = None
    auth_expires_at: int | None = None
    github_oauth_token: str = ""
    github_token_type: str = "bearer"
    github_scope: str = ""
    github_user_login: str = ""
    github_user_id: int | None = None
    copilot_access_token: str = ""
    copilot_token_expires_at: int | None = None

    _device_sessions: dict[str, DeviceAuthorizationSession] = PrivateAttr(
        default_factory=dict,
    )

    def _copilot_api_headers(self) -> dict[str, str]:
        return {
            "Editor-Version": "vscode/1.98.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": "CoPaw/0.2",
            "Accept": "application/json",
        }

    def _github_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "CoPaw/0.2",
        }

    def _oauth_auth_headers(self) -> dict[str, str]:
        headers = self._github_headers()
        headers["Authorization"] = f"Bearer {self.github_oauth_token}"
        return headers

    def _copilot_token_valid(self) -> bool:
        expires_at = self.copilot_token_expires_at or 0
        return bool(
            self.copilot_access_token
            and expires_at > int(time.time()) + COPILOT_TOKEN_REFRESH_SKEW,
        )

    def _sync_client(self, timeout: float = 10) -> httpx.Client:
        return httpx.Client(timeout=timeout, follow_redirects=True)

    def _client(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.copilot_access_token or self.api_key,
            timeout=timeout,
            default_headers=self._copilot_api_headers(),
        )

    async def start_device_authorization(
        self,
        timeout: float = 10,
    ) -> DeviceAuthorizationSession:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                GITHUB_DEVICE_CODE_URL,
                headers=self._github_headers(),
                data={
                    "client_id": GITHUB_COPILOT_OAUTH_CLIENT_ID,
                    "scope": "read:user",
                },
            )
            response.raise_for_status()
            payload = response.json()

        session = DeviceAuthorizationSession(
            session_id=secrets.token_urlsafe(18),
            device_code=str(payload["device_code"]),
            user_code=str(payload["user_code"]),
            verification_uri=str(payload["verification_uri"]),
            expires_at=int(time.time()) + int(payload.get("expires_in", 900)),
            interval=int(payload.get("interval", 5)),
        )
        self._device_sessions[session.session_id] = session
        return session

    async def poll_device_authorization(
        self,
        session_id: str,
        timeout: float = 10,
    ) -> tuple[str, str]:
        session = self._device_sessions.get(session_id)
        if session is None:
            return "missing", "Authorization session not found"

        if session.expires_at <= int(time.time()):
            self._device_sessions.pop(session_id, None)
            return "expired", "Device code expired"

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                GITHUB_ACCESS_TOKEN_URL,
                headers=self._github_headers(),
                data={
                    "client_id": GITHUB_COPILOT_OAUTH_CLIENT_ID,
                    "device_code": session.device_code,
                    "grant_type": (
                        "urn:ietf:params:oauth:grant-type:device_code"
                    ),
                },
            )
            response.raise_for_status()
            payload = response.json()

        if "error" in payload:
            status, message = self._map_device_flow_error(payload)
            session.status = status
            session.last_message = message
            if status in {"expired", "denied", "error"}:
                self._device_sessions.pop(session_id, None)
            return status, message

        self.github_oauth_token = str(payload.get("access_token", ""))
        self.github_token_type = str(payload.get("token_type", "bearer"))
        self.github_scope = str(payload.get("scope", ""))
        self.is_authenticated = bool(self.github_oauth_token)
        await self._populate_github_user(timeout=timeout)
        await self._refresh_copilot_token_async(timeout=timeout)
        self._device_sessions.pop(session_id, None)
        return "authorized", "GitHub authorization completed"

    @staticmethod
    def _map_device_flow_error(payload: dict[str, Any]) -> tuple[str, str]:
        error = str(payload.get("error", ""))
        description = str(payload.get("error_description", "")).strip()
        message = description or error or "Unknown device authorization error"
        if error == "authorization_pending":
            return "pending", message
        if error == "slow_down":
            return "pending", message
        if error == "expired_token":
            return "expired", message
        if error == "access_denied":
            return "denied", message
        return "error", message

    async def _populate_github_user(self, timeout: float = 10) -> None:
        if not self.github_oauth_token:
            return
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                GITHUB_USER_URL,
                headers=self._oauth_auth_headers(),
            )
            response.raise_for_status()
            payload = response.json()
        self.github_user_login = str(payload.get("login", ""))
        user_id = payload.get("id")
        self.github_user_id = int(user_id) if user_id is not None else None
        self.auth_account_label = self.github_user_login or None

    async def _refresh_copilot_token_async(self, timeout: float = 10) -> None:
        if self._copilot_token_valid():
            self.api_key = self.copilot_access_token
            return
        if not self.github_oauth_token:
            raise ValueError("GitHub authorization required")

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                GITHUB_COPILOT_TOKEN_URL,
                headers={
                    **self._oauth_auth_headers(),
                    **self._copilot_api_headers(),
                },
            )
            response.raise_for_status()
            payload = response.json()

        self._apply_copilot_token_payload(payload)

    def _refresh_copilot_token_sync(self, timeout: float = 10) -> None:
        if self._copilot_token_valid():
            self.api_key = self.copilot_access_token
            return
        if not self.github_oauth_token:
            raise ValueError("GitHub authorization required")

        with self._sync_client(timeout=timeout) as client:
            response = client.get(
                GITHUB_COPILOT_TOKEN_URL,
                headers={
                    **self._oauth_auth_headers(),
                    **self._copilot_api_headers(),
                },
            )
            response.raise_for_status()
            payload = response.json()

        self._apply_copilot_token_payload(payload)

    def _apply_copilot_token_payload(self, payload: dict[str, Any]) -> None:
        token = str(payload.get("token", "") or payload.get("access_token", ""))
        if not token:
            raise ValueError("GitHub Copilot token exchange failed")
        expires_at = payload.get("expires_at")
        if expires_at is None:
            expires_at = int(time.time()) + DEFAULT_COPILOT_TOKEN_TTL
        self.copilot_access_token = token
        self.copilot_token_expires_at = int(expires_at)
        self.auth_expires_at = self.copilot_token_expires_at
        self.api_key = token

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        if not self.github_oauth_token:
            return False, "GitHub authorization required"
        try:
            await self._refresh_copilot_token_async(timeout=timeout)
            client = self._client(timeout=timeout)
            await client.models.list(timeout=timeout)
            return True, ""
        except APIError as exc:
            return False, f"API error when connecting to `{self.base_url}`: {exc}"
        except Exception as exc:
            return False, f"Unknown exception when connecting to `{self.base_url}`: {exc}"

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        if not self.github_oauth_token:
            return []
        try:
            await self._refresh_copilot_token_async(timeout=timeout)
            client = self._client(timeout=timeout)
            payload = await client.models.list(timeout=timeout)
            return self._normalize_models_payload(payload)
        except APIError:
            return []
        except Exception:
            return []

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        if not self.github_oauth_token:
            return False, "GitHub authorization required"
        await self._refresh_copilot_token_async(timeout=timeout)
        return await super().check_model_connection(model_id=model_id, timeout=timeout)

    def get_chat_model_instance(self, model_id: str):
        self._refresh_copilot_token_sync(timeout=10)
        return OpenAIChatModelCompat(
            model_name=model_id,
            stream=True,
            api_key=self.copilot_access_token,
            stream_tool_parsing=False,
            client_kwargs={
                "base_url": self.base_url,
                "default_headers": self._copilot_api_headers(),
            },
            generate_kwargs=self.generate_kwargs,
        )

    def logout(self) -> None:
        self.is_authenticated = False
        self.auth_account_label = None
        self.auth_expires_at = None
        self.github_oauth_token = ""
        self.github_token_type = "bearer"
        self.github_scope = ""
        self.github_user_login = ""
        self.github_user_id = None
        self.copilot_access_token = ""
        self.copilot_token_expires_at = None
        self.api_key = ""
        self._device_sessions.clear()

    async def get_info(self, mock_secret: bool = True) -> ProviderInfo:
        self.is_authenticated = bool(self.github_oauth_token)
        self.auth_account_label = self.github_user_login or None
        self.auth_expires_at = self.copilot_token_expires_at
        info = await super().get_info(mock_secret=mock_secret)
        info.api_key = ""
        return info