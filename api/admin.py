from django.contrib import admin, messages
from django.template.response import TemplateResponse
from django.urls import path
from django.utils.html import format_html

from .models import ApiClient, ApiUsageGuide


def token_message(client, raw_token, *, regenerated=False):
    heading = "New API token" if regenerated else "API token created"
    return format_html(
        "<strong>{} for {} ({})</strong><br>"
        "Copy this token now. It will not be shown again:<br>"
        '<code style="display:inline-block;margin:6px 6px 6px 0;'
        'padding:6px 8px;user-select:all">{}</code>'
        '<button type="button" class="button" data-token="{}" '
        "onclick=\"const b=this; navigator.clipboard.writeText(b.dataset.token)"
        ".then(()=>{{b.textContent='Copied!';setTimeout(()=>b.textContent='Copy token',1500)}})"
        ".catch(()=>{{const t=document.createElement('textarea');t.value=b.dataset.token;"
        "document.body.appendChild(t);t.select();document.execCommand('copy');t.remove();"
        "b.textContent='Copied!';setTimeout(()=>b.textContent='Copy token',1500)}})\">"
        "Copy token</button><br>"
        'See the <a href="/admin/api/apiusageguide/">API usage guide</a> for request examples.',
        heading,
        client.name,
        client.code,
        raw_token,
        raw_token,
    )


@admin.register(ApiClient)
class ApiClientAdmin(admin.ModelAdmin):
    change_list_template = "admin/api/apiclient/change_list.html"
    list_display = (
        "name",
        "code",
        "is_active",
        "allowed_endpoints_display",
        "expires_at",
        "last_used_at",
        "created_at",
    )
    list_filter = ("is_active",)
    search_fields = ("name", "code", "notes")
    readonly_fields = ("code", "token_hash", "created_at", "last_used_at")
    fieldsets = (
        (None, {"fields": ("name", "code", "is_active", "expires_at", "notes")}),
        ("Access", {"fields": ("allowed_endpoints",)}),
        ("Security", {"fields": ("token_hash", "created_at", "last_used_at")}),
    )

    def allowed_endpoints_display(self, obj):
        if not obj.allowed_endpoints:
            return "all"
        return ", ".join(obj.allowed_endpoints)

    allowed_endpoints_display.short_description = "Allowed endpoints"

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["api_guide_url"] = "/admin/api/apiusageguide/"
        return super().changelist_view(request, extra_context=extra_context)

    def save_model(self, request, obj, form, change):
        if not change:
            raw_token = ApiClient.generate_token()
            obj.code = ApiClient.generate_code()
            obj.set_token(raw_token)
            super().save_model(request, obj, form, change)
            messages.warning(request, token_message(obj, raw_token))
            return
        super().save_model(request, obj, form, change)

    @admin.action(description="Regenerate token (shown once)")
    def regenerate_token(self, request, queryset):
        for client in queryset:
            raw_token = ApiClient.generate_token()
            client.set_token(raw_token)
            client.save(update_fields=["token_hash"])
            messages.warning(
                request,
                token_message(client, raw_token, regenerated=True),
            )

    actions = ["regenerate_token"]


@admin.register(ApiUsageGuide)
class ApiUsageGuideAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return False

    def has_module_permission(self, request):
        return request.user.is_staff

    def get_urls(self):
        info = self.model._meta.app_label, self.model._meta.model_name
        return [
            path(
                "",
                self.admin_site.admin_view(self.guide_view),
                name="%s_%s_changelist" % info,
            ),
        ]

    def guide_view(self, request):
        context = {
            **self.admin_site.each_context(request),
            "title": "FaceID API Usage Guide",
            "opts": self.model._meta,
        }
        return TemplateResponse(request, "admin/api_usage_guide.html", context)
