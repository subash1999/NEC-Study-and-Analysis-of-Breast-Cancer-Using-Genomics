from django.contrib import admin
from django.urls import path, include
from .import views

app_name="predict"

urlpatterns = [
    path('index', views.index ,name='index'),
    path('', views.welcome ,name='welcome'),
    path('selected_model/<int:index>/<int:row_index>', views.selectedModel ,name='selected_model'),
]
