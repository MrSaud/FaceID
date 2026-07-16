from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("detect/", views.detect_face, name="detect"),
    path("liveness/challenge/", views.create_challenge, name="liveness-challenge"),
    path("liveness/speech/", views.detect_speech_liveness, name="liveness-speech"),
    path("liveness/", views.detect_liveness, name="liveness"),
    path("compare/", views.compare_faces, name="compare"),
]
