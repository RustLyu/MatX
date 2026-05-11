#include <windows.h>
#include <stdio.h>

#define DIM 2

typedef char      f77_char;
typedef int       f77_int;
typedef float     f77_float;
typedef double    f77_double;

typedef char* (*Fptr_Version)(void);
// Function pointer declaration for dgemm api
typedef void(*Fptr_dgemm)(const f77_char* N1, const f77_char* N2, const f77_int* M, const f77_int* N,
	const f77_int* K, const double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb,
	const double* beta, const double* c, const f77_int* ldc);

int main()
{
	HMODULE hModule0, hModule1;
	double a[DIM * DIM] = { 1.0, 3.0, 2.0, 4.0 };
	double b[DIM * DIM] = { 5.0, 7.0, 6.0, 8.0 };
	double c[DIM * DIM];
	int I, J, M, N, K, lda, ldb, ldc;
	double alpha, beta;
	M = DIM;
	N = M;
	K = M;
	lda = M;
	ldb = K;
	ldc = M;
	alpha = 1.0;
	beta = 0.0;
    
	hModule0 = LoadLibrary(TEXT("..\\lib\\libomp.dll"));
	hModule1 = LoadLibrary(TEXT("..\\lib\\AOCL-LibBlis-Win-MT-dll.dll"));
	if (NULL == hModule1)
	{
		printf("Load Library failed \n");
		return -1;
	}
	printf("Load Library successfully loaded \n");

	Fptr_Version BLIS_VER = (Fptr_Version)GetProcAddress(hModule1, "bli_info_get_version_str");
	if (NULL == BLIS_VER)
	{
		printf("Function address not valid \n");
		return -1;
	}
	printf("\n############################################################\n \n");
	printf("             Blis Version %s \n", BLIS_VER());
	printf("\n############################################################ \n");
	printf("a = \n");
	for (I = 0; I < M; I++)
	{
		for (J = 0; J < K; J++)
		{
			printf("%f\t", a[J * K + I]);
		}
		printf("\n");
	}
	printf("b = \n");
	for (I = 0; I < K; I++)
	{
		for (J = 0; J < N; J++)
		{
			printf("%f\t", b[J * N + I]);
		}
		printf("\n");
	}

	Fptr_dgemm DGEMM = (Fptr_dgemm)GetProcAddress(hModule1, "dgemm");
	if (NULL == DGEMM)
	{
		printf("Function address not valid \n");
        return -1;
	}
	printf("Addition Function mapped  \n");

	DGEMM("N", "N", &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
	printf("c = \n");
	for (I = 0; I < M; I++)
	{
		for (J = 0; J < N; J++)
		{
			printf("%f\t", c[J * N + I]);
		}
		printf("\n");
	}
    FreeLibrary(hModule0);
	FreeLibrary(hModule1);
	return 0;
}