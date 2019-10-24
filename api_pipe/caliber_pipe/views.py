
# from caliber_pipe.models import Pipe
# from caliber_pipe.serializers import PipeSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .pipe_detect import opencv_pipe_detect

@api_view(['POST'])
def pipe_caliber(request):
    if request.method == 'POST':
        image_root = request.data['path']
        distance = opencv_pipe_detect(image_root)   
        return Response({'caliber_pixle': distance})
