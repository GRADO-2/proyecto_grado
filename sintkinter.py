import cv2 
import numpy as np
import mediapipe as mp 

width, height = 600, 400
manosdetector = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
widthv = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
heightv = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

numero_dato = 0


datos = []
with manosdetector.Hands(min_detection_confidence = 0.8 , min_tracking_confidence= 0.5) as hands:
    # bucle de video
        #toma un frame de la webcam
    while(True):
        tru, frame = video.read()

        #cambiamos de color al frame para poder tomar las caracteristicas y mandarlo a prediccion
        imagen = cv2.flip(frame,1)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        # hacemos que no sea modificable .necesario.
        imagen.flags.writeable = False
        # pasamos la imagen al modelo de prediccion
        prediction = hands.process(imagen)
        imagen.flags.writeable = True
        # cambiamos la imagen a su color original
        imagenT = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        # si detecto marcas de manos, iremos por cada cordenada que encontro pasandola a la imagen
        coorde_x , coorde_y = [],[]
        if prediction.multi_hand_landmarks:
            for num,hand in enumerate(prediction.multi_hand_landmarks):
                mp_drawing.draw_landmarks(imagenT, hand, manosdetector.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                for posicion in manosdetector.HandLandmark:
                    coorde_x.append(int(hand.landmark[posicion].x * imagenT.shape[1]))
                    coorde_y.append(int(hand.landmark[posicion].y * imagenT.shape[0]))
            print(imagenT.shape)
            if((min(coorde_x) - 20) < 0 or (min(coorde_y) - 20) < 0):
                cv2.rectangle(imagenT,(0 , max(coorde_y) + 20),(max(coorde_x) + 20, min(coorde_y) - 20),(255,255,255), 5)
                imagenT = imagenT[min(coorde_y) - 20 : max(coorde_y) +  20 ,0: max(coorde_x) + 20]
            else:
                cv2.rectangle(imagenT,(min(coorde_x) - 20, max(coorde_y) + 20),(max(coorde_x) + 20, min(coorde_y) - 20),(255,255,255), 5)
                imagenT = imagenT[min(coorde_y) - 20 : max(coorde_y) +  20 ,min(coorde_x) - 20: max(coorde_x) + 20]

            # la imagen al fin es enseñada en pantalla

        cv2.imshow("VIDEO", imagenT)

        if cv2.waitKey(10) & 0xFF == ord("a"):
            print("cargando imagenes...")

            datos.append(imagenT)
            
        if cv2.waitKey(10) & 0xFF == ord("s"):
            print("restart..")
            videosubir = cv2.VideoWriter('./data_samples/dato{}.avi'.format(numero_dato),cv2.VideoWriter_fourcc(*'XVID'),20.0,(widthv,heightv))
            for x in datos:
                
                videosubir.write(x)
            
            videosubir.release()
            print("HECHO..")
            datos = []
            numero_dato += 1
        if cv2.waitKey(10) & 0xFF == ord("x"):
            print("stop.")
            break

        
        
video.release()
cv2.destroyAllWindows()
