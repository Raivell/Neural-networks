from imageai.Detection import ObjectDetection
import os
from imageai.Prediction import ImagePrediction


exec_path = os.getcwd()
"""
 person,   bicycle,   car,   motorcycle,   airplane,
 bus,   train,   truck,   boat,   traffic light,   fire hydrant,   stop_sign,
 parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
 giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
 sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
 bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
 broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
 dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
 oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
 toothbrush
"""
#создаем класс 
detector = ObjectDetection()
#выполн. задачи обнаружения объекта, используя предварительно обученную модель «RetinaNet»
detector.setModelTypeAsRetinaNet()
# указываю путь к загруженному файлу модели
detector.setModelPath(os.path.join(
    exec_path, "resnet50_coco_best_v2.0.1.h5"))
#функция загружает модель из пути, указанного в вызове функции выше, в экземпляр обнаружения объекта
detector.loadModel()

#функция, которая выполняет задачу обнаружения объекта после загрузки модели
list = detector.detectObjectsFromImage(
    input_image=os.path.join(exec_path, "1.jpg"),
	output_image_path=os.path.join(exec_path, "new_objects.jpg"),
	minimum_percentage_probability=60,
	display_percentage_probability=True,
	display_object_name=True,
	#извлекает и сохраняет каждый обьект обнаруженный на изображении
	extract_detected_objects=False
	)

#Классы прогнозирования
prediction = ImagePrediction ()
prediction . setModelTypeAsResNet ()
prediction.setModelPath(os.path.join(
    exec_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction . loadModel ()

predictions , probabilities = prediction . predictImage( os . path . join (exec_path , "Картинка.jpg"), result_count = 5 )
for eachPrediction , eachProbability in zip ( predictions , probabilities ):
    print ( eachPrediction , " : " , eachProbability )
