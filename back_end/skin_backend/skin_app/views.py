from django.shortcuts import render
from django.http import HttpResponse
from skin_app.forms import Skin_recogform
from skin_app.model_prediction import pipeline_model,extract_details
from django.conf import settings
from skin_app.models import Skindieases
import os

# Create your views here.

def index(request):
    form= Skin_recogform()

    if request.method == 'POST':
        form = Skin_recogform(request.POST or None, request.FILES or None )
        if form.is_valid():
            save = form.save(commit=True)
       
            #extract the image object from database 
            primary_key = save.pk
            imageobj = Skindieases.objects.get(pk=primary_key)
            fileroot = str(imageobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT,fileroot)      
            results = pipeline_model(filepath)
            print(results)
            gemini_resp=extract_details(results)
          #  print(gemini_resp)

            return render(request,'index.html',{'form':form,'upload':True,'resultS':results,'ext_detail':gemini_resp})


    return render(request,'index.html',{'form':form,'upload':False})