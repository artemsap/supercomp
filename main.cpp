#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <ctime>

#define THREADS 3
#define MAX_NUM 100
#define MIN_NUM -100

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C);
void print_matrix(int num_col, int num_str, double *matrix);
void generate_matrix(int num_col, int num_str, double *matrix);

int main (void)
{
  std::srand(time(0));
  int max_num = 5000;

  //int M = std::rand()%max_num + 1; //Количество столбцов в матрице А и строк в матрице B
  //int N = std::rand()%max_num + 1; //Количество строк в матрице А
  //int K = std::rand()%max_num + 1; //Количество столбцов в матрице B

  int M = 1000;
  int N = 2000;
  int K = 1500;

  double *A = new double[M*N];
  double *B = new double[M*K];
  double *C = new double[N*K];

  double start = clock(); // засекаем время старта

  generate_matrix(M, N, A);
  generate_matrix(K, M, B);

  double end = clock(); // засекаем время окончания

  double t1 = (end - start) / CLOCKS_PER_SEC;

  //printf("MATRIX A\n");
  //print_matrix(M, N, A);

  //printf("MATRIX B\n");
  //print_matrix(K, M, B);

  memset(C, 0, sizeof(double)*N*K);

  start = clock(); // засекаем время старта

  blas_dgemm(M, N, K, A, B, C);

  end = clock(); // засекаем время окончания
  
  double t = (end - start) / CLOCKS_PER_SEC;

  //printf("MATRIX C\n");
  //print_matrix(K, N, C);

  printf("Time of matrix generation=%f \n", t1);
  printf("Time of matrix multiplication=%f \n", t);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C)
{
  #pragma omp parallel num_threads(THREADS)
  {
    #pragma omp for
    for(int z=0; z<N; z++)
      for(int j=0; j<K; j++)
        for(int i=0; i<M; i++)
          C[z*K+j] += A[z*M+i]*B[i*K+j];
  }
}

void print_matrix(int num_col, int num_str, double *matrix)
{
  for(int i=0; i<num_str; i++)
  {
    for(int j=0; j<num_col; j++)
      printf("%5.2f \t", matrix[i*num_col +j]);
    printf("\n");
  }
}

void generate_matrix(int num_col, int num_str, double *matrix)
{
  #pragma omp parallel num_threads(THREADS)
  {
    #pragma omp for
    for(int i=0; i<num_col; i++)
      for(int j=0; j<num_str; j++)
        matrix[i*num_str +j] = (MIN_NUM + std::rand()%(MAX_NUM-MIN_NUM+1))/100.0;
  }
}
