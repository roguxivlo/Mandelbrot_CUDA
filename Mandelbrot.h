//
//  Mandelbrot.h
//  
//
//  Created by Witold Rudnicki 
//
//

#define real float

#ifndef ____Mandelbrot__
#define ____Mandelbrot__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#endif /* defined(____Mandelbrot__) */

int computeMandelbrot(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER, int *Mandel);
int computeMandelbrot2D(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER, int *Mandel);

void makePicture(int *Mandel, int width, int height, int MAX);
void makePicturePNG(int *Mandel,int width, int height, int MAX, int id);