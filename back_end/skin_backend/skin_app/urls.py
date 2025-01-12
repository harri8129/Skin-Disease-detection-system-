from django.urls import path
from skin_app import views

urlpatterns = [
    path('',views.index,name='html_function')
    
]