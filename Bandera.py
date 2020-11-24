import numpy as np
import cv2
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

#Librerias para el método de orientacion
from hough import hough
from Sobel_orientation_estimate import *


class Bandera:
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen #Guarda la ruta de la imagen en la variable ruta_imagen.
        self.image=cv2.imread(self.ruta_imagen) #Lee la imagen según la ruta correspondiente y la guarda en la variable image.
        #self.imageRGB= np.copy(self.image) #Realiza una copia de la imagen.
        #cv2.imshow('bandera', self.imagen)#Muestra la imagen gris en la pantalla.
        #cv2.waitKey(0)

    def Colores(self):

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        n_colors = 4
        image = np.array(self.image, dtype=np.float64) / 255  # Normalización.

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape  # Dimensiones de la imagen
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))  # Nueva matriz.

        image_array_sample = shuffle(image_array, random_state=0)[:10000]  # Se selecionan mil pixeles.

        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)  #

        self.labels = model.predict(image_array) #Labels
        centers = model.cluster_centers_  # Centro de los clusters.

        x = set(self.labels)
        len(x)
        print(len(x))

        # # Display all results, alongside original image
        # plt.figure(1)
        # plt.clf()
        # plt.axis('off')
        # plt.title('Original image')
        # plt.show(image)

    def Porcentaje (self):

        vector= [0,0,0,0]

        for i in self.labels:
            if i == 1:
                vector[0] = vector[0]+1
            elif i == 2:
                vector[1] = vector[1]+1
            elif i == 3:
                vector[2] = vector[2]+1
            else:
                vector[3] = vector[3]+1

        porcentaje_0 = (vector[0]/len(self.labels))*100
        if (porcentaje_0 != 0) :
            print(porcentaje_0)

        porcentaje_1 = (vector[1]/len(self.labels))*100
        if porcentaje_1 != 0:
            print(porcentaje_1)

        porcentaje_2 = (vector[2]/len(self.labels))*100
        if porcentaje_2 != 0:
            print(porcentaje_2)

        porcentaje_3 = (vector[3]/len(self.labels))*100
        if porcentaje_3 != 0:
            print(porcentaje_3)

        lista = [porcentaje_0,porcentaje_1,porcentaje_2,porcentaje_3]
        #print(lista)

    def Orientacion(self):

         high_thresh = 300
         bw_edges = cv2.Canny(self.image, high_thresh * 0.3, high_thresh, L2gradient=True)
         # print(bw_edges)

         hough = hough(bw_edges)

         accumulator = hough.standard_HT()

         acc_thresh = 50
         N_peaks = 11
         nhood = [25, 9]
         peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

         [_, cols] = image.shape[:2]
         image_draw = np.copy(image)
         for i in range(len(peaks)):
             rho = peaks[i][0]
             theta_ = hough.theta[peaks[i][1]]

             theta_pi = np.pi * theta_ / 180
             theta_ = theta_ - 180
             a = np.cos(theta_pi)
             b = np.sin(theta_pi)
             x0 = a * rho + hough.center_x
             y0 = b * rho + hough.center_y
             c = -rho
             x1 = int(round(x0 + cols * (-b)))
             y1 = int(round(y0 + cols * a))
             x2 = int(round(x0 - cols * (-b)))
             y2 = int(round(y0 - cols * a))

             if np.abs(theta_) < 80:
                 image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
             elif np.abs(theta_) > 100:
                 image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
             else:
                 if theta_ > 0:
                     image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                 else:
                     image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

         # cv2.imshow("frame", bw_edges)
         # cv2.imshow("lines", image_draw)
         # cv2.waitKey(0)

         row, cols = bw_edges.shape
         print(row)
         print(cols)

         pixel_actual = 0
         pixel_anterior = 0
         vertical = 0

         for r in range(row):
             pixel_actual = 1
             pixel_anterior = 1
             for c in range(cols):
                 pixel_actual = bw_edges[r][c]
                 if pixel_actual == 255 and pixel_anterior == 0:
                     vertical = 1
                 pixel_anterior = pixel_actual

         horizontal = 0
         for c in range(cols):
             pixel_actual = 1
             pixel_anterior = 1
             for r in range(row):
                 pixel_actual = bw_edges[r][c]
                 if pixel_actual == 255 and pixel_anterior == 0:
                     horizontal = 1
                 pixel_anterior = pixel_actual
         if horizontal == 1 and vertical == 1:
             print('Bandera mixta')
         elif horizontal == 1:
             print('Bandera horizontal')
         elif vertical == 1:
             print('Bandera vertical')




