// compilación: nvcc -o prueba sobel-cuda.cpp sobel.cu


#define cimg_display 0
#include "CImg.h"
#include <iostream>
#include <cmath>
#include <time.h>
#include "sobel.h"

using namespace cimg_library;
using namespace std;


int main(int argc, char **argv){

   int filas, columnas;
   float *datosImagenTemp;
   double t;
   struct timespec cgt1,cgt2;
   double ncgt; //para tiempo de ejecución
   
   if (argc != 2){
      cout<<"\nERROR: falta el nombre de la imagen de entrada\n";
      return -1;
   }

   CImg <float> imagenEntrada(argv[1]);
   filas= imagenEntrada.height();
   columnas= imagenEntrada.width();
   
   // Reservamos espacio para los datos de la imagen solución
   datosImagenTemp= new float[filas*columnas*3];
   
   // Tomamos primer tiempo
   clock_gettime(CLOCK_REALTIME,&cgt1);
   
   calcularSobelCuda (imagenEntrada.data(), datosImagenTemp, filas, columnas);
   
   // Tomamos segundo tiempo
   clock_gettime(CLOCK_REALTIME,&cgt2);   
   // Tiempo de ejecución
   ncgt=(double) (cgt2.tv_sec-cgt1.tv_sec)+ (double) ((cgt2.tv_nsec-cgt1.tv_nsec)/(1.e+9));
   
   // Creamos la imagen de salida a partir de los datos calculados
   CImg <float> imagenSalida(datosImagenTemp, columnas, filas, 1, 3);
   
   // Mostramos resultados
   imagenSalida.save_jpeg("resultado.jpg");
   cout<<"\nTiempo de ejecución: "<<ncgt<<endl;
   
   return 0;
}
