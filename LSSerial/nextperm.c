#define STACKSIZE 32768    /* tamanho de pilha das threads */
#define _XOPEN_SOURCE 600  /* para compilar no MacOS */


#include <stddef.h>
#include <stdio.h>
#include "LIS.c"
#include "LDS.c"
#include <time.h>
#include "EnumaratorSequence.cu"


void inversion(int *array, int length){
	int i;
	for(i = 0; i < length/2; i++){
		int temp = array[i];
		array[i] = array[length - i-1];
		array[length-i-1] = temp;
	}
}

void rotation(int *array, int length){
  int i, temp;
  temp = array[0];
  for (i = 0; i < length-1; i++)
     array[i] = array[i+1];
  array[i] = temp;
}

int next_permutation(int *array, size_t length) {
	size_t i, j;
	int temp;
	// Find non-increasing suffix
	if (length == 0)
		return 0;
	i = length - 1;
	while (i > 0 && array[i - 1] >= array[i])
		i--;
	if (i == 0)
		return 0;
	
	// Find successor to pivot
	j = length - 1;
	while (array[j] <= array[i - 1])
		j--;
	temp = array[i - 1];
	array[i - 1] = array[j];
	array[j] = temp;
	
	// Reverse suffix
	j = length - 1;
	while (i < j) {
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
		i++;
		j--;
	}
	return 1;
}

void printVector(int* array, int length){
	int k;
	for(k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

//Min(|LIS(s)|, |LDS(s)|)

void decideLS(int *vector, int length, int* lmin){
	
	unsigned int lLIS, lLDS, latual; 
	
	lLIS = LIS(vector, length);
	lLDS = LDS(vector, length);
	
	//printf("lLIS = %d, lLDS = %d , Lmin = %d\n",lLIS, lLDS,  *lmin);

	if(lLIS < *lmin){
		
		*lmin = lLIS;
	}

	if(lLDS < *lmin){
		*lmin = lLDS;	
	}
	
}

//Seja S o conjunto de todas las sequencias dos n primeiros números naturais.
//Defina R(s), com s \in S o conjunto de todas as sequencias que podem
//ser geradas rotacionando S.
//Defina LIS(s) e LDS(s) como você sabe e sejam |LIS(s)| e |LDS(s)| suas
//cardinalidades.
//Determinar Max_{s \in S}(Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)))


int main(int argc, char* argv[]){
	int* vector;
	int* vecRotation;
	int* vecInversion;	
	int length = atoi(argv[1]);
	int lR_expected = atoi(argv[2]);
	int num_expected = 0;
	clock_t start,end;

	printf("%lu\n", sizeof(int));
	printf("%lu\n", sizeof(long));
	printf("%lu\n", sizeof(long long));
	
	vector = (int*) malloc(sizeof(int)*length);
	vecRotation = (int*) malloc(sizeof(int)*length);
	vecInversion = (int*) malloc(sizeof(int)*length);

	start = clock();
	
	int i;
	for(i = 0; i < length; i++)
		vector[i] = i+1;

	unsigned int lmaxS = 0;
	//Length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto.
	int counter = fatorial(length-1)/2;
	printf("counter: %d\n",counter);

        //Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
	//
	while(counter){
		unsigned int lminR = length;	
		//printf("Permutação: ");
		//printVector(vector,length);
		decideLS(vector, length, &lminR);

		memcpy(vecInversion,vector, sizeof(int)*length);
		inversion(vecInversion, length);
		//printf("Inversão: ");
		//printVector(vecInversion,length);
		decideLS(vecInversion, length, &lminR);

		memcpy(vecRotation,vector, sizeof(int)*length);
		for(i = 0; i < length-1; i++){
			rotation(vecRotation, length);
			//printf("Rotações: ");
			//printVector(vecRotation, length);
			decideLS(vecRotation, length, &lminR);


			memcpy(vecInversion,vecRotation, sizeof(int)*length);
			inversion(vecInversion, length);
			//printf("Inversão: ");
			//printVector(vecInversion,length);
			decideLS(vecInversion, length, &lminR);
		}

		if(lminR == lR_expected){
			num_expected++;
			printf("Id: %d ---", getIndex(vector,length));
			printVector(vector, length);
		}
		//Define o maior valor encontrado entre os elementos de S
		if(lmaxS < lminR){
			lmaxS = lminR;
		}

		next_permutation(vector+1,length-1);
		//printf("\n");
		counter--;
	}
	end = clock();
	printf("Número encontrado: %d\n",num_expected);
	printf("Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);
	printf("Lmax R = %d\n",lmaxS);
	free(vector);
	free(vecInversion);
	free(vecRotation);
}
