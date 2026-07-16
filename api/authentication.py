from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from .models import ApiClient


def _extract_bearer_token(authorization: str) -> str:
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def resolve_api_endpoint(request) -> str:
    path = request.path.rstrip("/")
    if path.endswith("/health"):
        return "health"
    if path.endswith("/detect"):
        return "detect"
    if path.endswith("/compare"):
        return "compare"
    if "/liveness" in path:
        return "liveness"
    return "unknown"


class ClientCodeTokenAuthentication(BaseAuthentication):
    """
    Expects:
      X-Client-Code: CD82
      Authorization: Bearer fid_live_...
    """

    www_authenticate_realm = "FaceID API"

    def authenticate(self, request):
        code = str(request.headers.get("X-Client-Code") or "").strip().upper()
        token = _extract_bearer_token(request.headers.get("Authorization", ""))

        if not code or not token:
            raise AuthenticationFailed("X-Client-Code and Authorization Bearer token are required")

        try:
            client = ApiClient.objects.get(code=code, is_active=True)
        except ApiClient.DoesNotExist as exc:
            raise AuthenticationFailed("Invalid client code or token") from exc

        if client.is_expired():
            raise AuthenticationFailed("API client has expired")

        if not client.check_token(token):
            raise AuthenticationFailed("Invalid client code or token")

        endpoint = resolve_api_endpoint(request)
        if not client.allows_endpoint(endpoint):
            raise AuthenticationFailed(f"Client is not allowed to access '{endpoint}'")

        client.mark_used()
        return client, token
