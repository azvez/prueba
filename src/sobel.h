//
// Archivo de cabecera para sobel.cu
//



// Kernel que se ejecutará en el grid de la GPU
//__global__ void sobelKernel(float *entrada, float *salida, int filas, int columnas);

// Función que lanza la ejecución de vectores en la GPU
void calcularSobelCuda (float *hEntrada, float *hSalida, int filas, int columnas);
