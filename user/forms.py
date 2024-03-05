from django.forms import forms
from . import models
class UserInput(forms.Form):
    class Meta:
        model = models.PredictCost
        fields = '__all__'