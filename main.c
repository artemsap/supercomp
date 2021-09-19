#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#define THREADS 10
#define MAX_NUM 1000
#define MIN_NUM -1000
#define MAX_SIZE_MATRIX 500

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C);
void print_matrix(int num_col, int num_rows, double *matrix);
void generate_matrix(int num_col, int num_rows, double *matrix);

int main (void)
{  
    int M = rand()%MAX_SIZE_MATRIX + 1; //Number of columns in A matrix and rows in B matrix
    int N = rand()%MAX_SIZE_MATRIX + 1; //Number of rows in A matrix
    int K = rand()%MAX_SIZE_MATRIX + 1; //Number of columns in A matrix

    double *A = malloc(sizeof(double)*M*N);
    double *B = malloc(sizeof(double)*M*K);
    double *C = malloc(sizeof(double)*K*N);

    memset(C, 0, sizeof(double)*N*K);

    // Time before matrix generation
    double start = clock();

    //Matrix generation
    generate_matrix(M, N, A);
    generate_matrix(K, M, B);

    // Time after matrix generation
    double end = clock();

    // Total time of matrix generation
    double t1 = (end - start) / CLOCKS_PER_SEC;

    // Time before matrix multiplication
    start = clock();

    // Matrix multiplications
    blas_dgemm(M, N, K, A, B, C);

    // Time after matrix multiplication
    end = clock();

    // Total time of matrix multiplication
    double t = (end - start) / CLOCKS_PER_SEC;

    printf("Time of matrix generation=%f \n", t1);
    printf("Time of matrix multiplication=%f \n", t);

    free(A);
    free(B);
    free(C);

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

void print_matrix(int num_col, int num_rows, double *matrix)
{
    for(int i=0; i<num_rows; i++)
    {
        for(int j=0; j<num_col; j++)
            printf("%5.2f \t", matrix[i*num_col +j]);
        printf("\n");
    }
}

void generate_matrix(int num_col, int num_rows, double *matrix)
{
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for(int i=0; i<num_col; i++)
            for(int j=0; j<num_rows; j++)
                matrix[i*num_rows +j] = (MIN_NUM + rand()%(MAX_NUM-MIN_NUM+1))/100.0;
    }
}
