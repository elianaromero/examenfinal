from Bandera import *
import cv2
import os

if __name__=='__main__':

    ruta = 'C:/Users/ASUS-PC/Desktop'
    nombre_imagen = input('Introduzca el nombre de la bandera: ')
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    image = cv2.imread(ruta_imagen)
    image = Bandera(ruta_imagen)
    image.Colores()
    image.Porcentaje()
    image.Orientacion()

