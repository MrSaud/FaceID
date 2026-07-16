from rest_framework.permissions import BasePermission

from .models import ApiClient


class IsApiClient(BasePermission):
    def has_permission(self, request, view):
        return isinstance(getattr(request, "user", None), ApiClient) and request.user.is_active
