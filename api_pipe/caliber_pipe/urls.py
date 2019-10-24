from django.urls import path
from caliber_pipe import views

urlpatterns = [
    path('mv-image-analysis/pipe/caliber', views.pipe_caliber),
]