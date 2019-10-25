
# from caliber_pipe.models import Pipe
# from caliber_pipe.serializers import PipeSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .pipe_detect import opencv_pipe_detect
from .service_detect_pipe import service_remove_noise
@api_view(['POST'])
def pipe_caliber(request):
    if request.method == 'POST':
        image_root = request.data['path']
        distance = opencv_pipe_detect(image_root)   
        return Response({'caliber_pixle': distance})
@api_view(['POST'])
def pipe_depth_cal(request):
    if request.method == 'POST':
        img_root = request.data['path']  
        actual_external_diameter = request.data['caliber']  
        depth,curve = service_remove_noise(img_root, actual_external_diameter)   
        return Response(
            {
                'depth': depth,
                'type': curve[0],
                'degree':curve[1]
                }
            )
