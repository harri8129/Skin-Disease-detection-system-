from django import forms 
from skin_app.models import Skindieases

class Skin_recogform(forms.ModelForm):

    class Meta:
        model = Skindieases
        fields = ['image']