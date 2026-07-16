import secrets
import string

from django.contrib.auth.hashers import check_password, make_password
from django.db import models
from django.utils import timezone


def _random_code(length: int = 4) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class ApiClient(models.Model):
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=16, unique=True, db_index=True)
    token_hash = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    allowed_endpoints = models.JSONField(
        blank=True,
        default=list,
        help_text='Leave empty to allow all endpoints. Example: ["detect", "liveness", "compare", "health"]',
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.code})"

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def is_expired(self) -> bool:
        return self.expires_at is not None and timezone.now() >= self.expires_at

    def set_token(self, raw_token: str) -> None:
        self.token_hash = make_password(raw_token)

    def check_token(self, raw_token: str) -> bool:
        return check_password(raw_token, self.token_hash)

    def allows_endpoint(self, endpoint: str) -> bool:
        allowed = self.allowed_endpoints or []
        if not allowed:
            return True
        return endpoint in allowed

    def mark_used(self) -> None:
        self.last_used_at = timezone.now()
        self.save(update_fields=["last_used_at"])

    @classmethod
    def generate_code(cls, length: int = 4) -> str:
        while True:
            code = _random_code(length)
            if not cls.objects.filter(code=code).exists():
                return code

    @classmethod
    def generate_token(cls) -> str:
        return f"fid_live_{secrets.token_urlsafe(32)}"

    @classmethod
    def create_client(cls, *, name: str, allowed_endpoints=None, expires_at=None, notes=""):
        raw_token = cls.generate_token()
        client = cls(
            name=name,
            code=cls.generate_code(),
            allowed_endpoints=allowed_endpoints or [],
            expires_at=expires_at,
            notes=notes,
        )
        client.set_token(raw_token)
        client.save()
        return client, raw_token


class ApiUsageGuide(models.Model):
    """Read-only admin entry that opens the API usage guide."""

    class Meta:
        managed = False
        verbose_name = "API usage guide"
        verbose_name_plural = "API usage guide"
