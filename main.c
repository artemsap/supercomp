#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#define THREADS 10
#define MAX_NUM 10000
#define MIN_NUM -10000
#define MAX_SIZE_MATRIX 5

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C, double alpha, double beta);
void print_matrix(int num_col, int num_rows, double *matrix);
void generate_matrix(int num_col, int num_rows, double *matrix);
void save_matrix_to_file(int num_col, int num_rows, double *matrix);

int main (void)
{  
    srand(time(0));

    int M = rand()%MAX_SIZE_MATRIX + 1; //Number of columns in A matrix and rows in B matrix
    int N = rand()%MAX_SIZE_MATRIX + 1; //Number of rows in A matrix
    int K = rand()%MAX_SIZE_MATRIX + 1; //Number of columns in B matrix

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
    blas_dgemm(M, N, K, A, B, C, 1, 0);

    // Time after matrix multiplication
    end = clock();

    // Total time of matrix multiplication
    double t = (end - start) / CLOCKS_PER_SEC;

    printf("Time of matrix generation=%f \n", t1);
    printf("Time of matrix multiplication=%f \n", t);

    printf("Matrix A\n");
    print_matrix(M, N, A);

    printf("Matrix B\n");
    print_matrix(K, M, B);

    printf("Matrix C\n");
    print_matrix(K, N, C);

    save_matrix_to_file(N, K, C);

    free(A);
    free(B);
    free(C);

    return 0;
}

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C, double alpha, double beta)
{
    #pragma omp parallel num_threads(THREADS)
    #pragma omp for
    for(int z=0; z<N; z++)
        for(int j=0; j<K; j++){
            double sum = 0;
            for(int i=0; i<M; i++)
                sum += A[z*M+i]*B[i*K+j];
            C[z*K+j] = beta*C[z*K+j] + alpha*sum;
        }
}

void print_matrix(int num_col, int num_rows, double *matrix)
{
    for(int i=0; i<num_rows; i++){
        for(int j=0; j<num_col; j++)
            printf("%5.2f \t", matrix[i*num_col +j]);
        printf("\n");
    }
}

void generate_matrix(int num_col, int num_rows, double *matrix)
{
    #pragma omp parallel num_threads(THREADS)
    #pragma omp for
    for(int i=0; i<num_col; i++)
        for(int j=0; j<num_rows; j++)
            matrix[i*num_rows +j] = (MIN_NUM + rand()%(MAX_NUM-MIN_NUM+1))/100.0;
}

void save_matrix_to_file(int num_col, int num_rows, double *matrix)
{
    FILE *file = fopen("matrix.txt", "w");
    
    for (int i = 0; i < num_col*num_rows; i++)
        fprintf(file, "%.2f ", matrix[i]);

    fclose(file);
}
