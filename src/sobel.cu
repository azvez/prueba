#define tamBloque 64

__constant__ int mascaraX[3][3];
__constant__ int mascaraY[3][3];


// Kernel que se ejecutará en el grid de la GPU
__global__ void sobelKernel(float *entrada, float *salida, int filas, int columnas){
   int sumx = 0;
   int sumy = 0;
   int SUM = 0;   
   int x, y;
   int i, j, pix, piy;
   int R, G, B, NC, posR, posG, posB;
   float newPixel;
   
   y= blockIdx.y;
   x= blockIdx.x * blockDim.x + threadIdx.x;

   if (y == 0 || y == filas-1 || x==0 || x == columnas-1){
      SUM = 0;
   }
   else{
      for(i=-1; i<=1; i++)  {
         for(j=-1; j<=1; j++)  {
            pix = j + x;
            piy = i + y;

            posR= piy*columnas + pix;  // posición en el vector del componente R del pixel sobre el que trabajamos
            R = (int)entrada[posR];  // imagen(pix,piy,0,0);
            
            posG= filas*columnas + piy*columnas + pix;  // posición en el vector del componente G del pixel sobre el que trabajamos
            G = (int)entrada[posG];  // imagen(pix,piy,0,1);
            
            posB= 2*filas*columnas + piy*columnas + pix;  // posición en el vector del componente B del pixel sobre el que trabajamos
            B = (int)entrada[posB];  // imagen(pix,piy,0,2);

            NC = (R+G+B)/3;

            sumx = sumx + (NC) * mascaraX[j+1][i+1];
            sumy = sumy + (NC) * mascaraY[j+1][i+1];

         }
      }
      SUM = abs(sumx) + abs(sumy);
   }
        
  if(SUM>255){
	 SUM=255;
  }
   
   newPixel = 255 - (float)(SUM);

   salida[y*columnas + x] = newPixel;  // componente R
   salida[filas*columnas + y*columnas + x] = newPixel;  // componente G
   salida[2*filas*columnas + y*columnas + x] = newPixel;  // componente B
}



// Función que lanza la ejecución de vectores en la GPU
void calcularSobelCuda (float *hEntrada, float *hSalida, int filas, int columnas){
   
   float *dEntrada, *dSalida;
   int tam;
   dim3 DimGrid, DimBlock;

   int Gx [3][3]; int Gy [3][3];
   // Sobel Horizontal Mask
   Gx[0][0] = 1; Gx[0][1] = 0; Gx[0][2] = -1;
   Gx[1][0] = 2; Gx[1][1] = 0; Gx[1][2] = -2;
   Gx[2][0] = 1; Gx[2][1] = 0; Gx[2][2] = -1;

   // Sobel Vertical Mask
   Gy[0][0] =  1; Gy[0][1] = 2; Gy[0][2] =   1;
   Gy[1][0] =  0; Gy[1][1] = 0; Gy[1][2] =   0;
   Gy[2][0] = -1; Gy[2][1] =-2; Gy[2][2] =  -1;

   // Transferimos las máscaras a la memoria constante de la GPU
   cudaMemcpyToSymbol(mascaraX, Gx, 3*3*sizeof(int));
   cudaMemcpyToSymbol(mascaraY, Gy, 3*3*sizeof(int));
   
   // Espacio que ocupa en memoria la imagen
   tam= filas * columnas * 3 * sizeof(float);  // 3 colores (R, G, B)
   
   // Reservamos espacio y copiamos en GPU la imagen de entrada
   cudaMalloc((void **) &dEntrada, tam);
   cudaMemcpy(dEntrada,hEntrada,tam,cudaMemcpyHostToDevice);
   
   // Reservamos espacio en GPU para la imagen de salida
   cudaMalloc((void **) &dSalida, tam);
   
   // tamaño del grid y de los bloques de hebras
   DimBlock= dim3(tamBloque, 1, 1);  // bloques de tamBloque hebras
   DimGrid= dim3( ((columnas-1)/tamBloque)+1, filas, 1); // grid 2D, x= bloques necesarios para cubrir 1 fila de la imagen, y= n filas imagen
      
   // Llamada al kernel
   sobelKernel<<<DimGrid,DimBlock>>>(dEntrada,dSalida,filas,columnas);

   // Copia de resultados GPU -> host
   cudaMemcpy(hSalida,dSalida,tam,cudaMemcpyDeviceToHost);
   
   // Liberación de memoria en GPU
   cudaFree(dEntrada);
   cudaFree(dSalida);
}
   
   
   
   
