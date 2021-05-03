import os

from django.shortcuts import render, HttpResponse, redirect
from django.conf import settings
from .models import *
from .forms import PostForm


def upload(request):
    if request.method == "POST":
        # print(request.POST)
        form = PostForm(request.POST, request.FILES)
        print(form.errors)
        if form.is_valid():
            form.save()
            return redirect(upload)
        # else:
        #     return HttpResponse("not valid")
        # print("*********************inside upload",request.FILES['file'])
    else:
        form = PostForm()
    d = request.FILES
    print('*************** file name =', d, '*************hosted at =', settings.ALLOWED_HOSTS[0] + ':8000')

    context = {'form': form, 'name': d, 'localhost': settings.ALLOWED_HOSTS[0] + ':8000'}
    return render(request, 'index.html', context)


def addData(request):
    name = Post.objects.all()
    dir1 = ''
    for i in name:
        dir1 = str(i)
    p = os.getcwd()
    p = p.replace("\\", "/")
    dir1 = p + "/media/" + dir1
    print("*****************DIR1 = ", dir1)

    Out_hr = HeartRateDetector(dir1)
    hrv = "http://127.0.0.1:8000/media/media/plot.png"
    vid = "http://127.0.0.1:8000/media/media/output.mp4v"
    print('out_hr =', Out_hr)
    print('hrv = ', hrv)
    print('vid =', vid)
    os.remove(dir1)
    # os.remove(hrv)
    # os.remove(vid)
    Post.objects.all().delete()
    return render(request, 'result advanced.html', {'heartrate': Out_hr, 'plot': hrv, 'video': vid})
