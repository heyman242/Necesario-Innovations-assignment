from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('stream/', views.live_stream, name='live_stream'),
]
