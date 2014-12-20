#include <stdio.h>
#include <stdlib.h>


/*
__device__
unsigned long long fatorial(int n){
	unsigned long long result = 1;
	int i;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}*/

//Coloca o elemento da variável pos na primeira posição, e da um shift para a direia nos outros
__device__
void shiftElement(char* dest, int pos){
	char temp = dest[pos];
	int i;
	for(i = 0; i < pos; i++){
		dest[pos-i] = dest[pos-i-1];
	}
	dest[0] = temp;
}

__device__
void getSequenceLexicographically(char* dest, int n, unsigned long long index){
	//Cria o vetor primário de tamanho N
	int i;
	for(i = 0; i < n; i++)
		dest[i] = i+4; //Começar do numero 4

	//Percorre o vetor
	for(i = 0; i < n-1; i++){
		//Calcula quantas alterações são possiveis sem alterar o primeiro elemento atual
		unsigned long long fat = tex1Dfetch(fatorial,n-i-1);
		//Calcula quantas vezes foi possível trocar essa posição
		int num_movimentos = index/fat;
		if(num_movimentos > 0){
			shiftElement(dest, num_movimentos);
			//Diminui a quantidade ja calcula do indice
			index -= num_movimentos*fat;
		}
		dest++;
	}
}

__device__
void getSequence(char* dest, int n, unsigned long long index){
	unsigned int numDeslocamentos2e3 = index/tex1Dfetch(fatorial,n-3);
	unsigned int indexResto = index%tex1Dfetch(fatorial,n-3);
	unsigned int pos_num2 = 1;
	unsigned int pos_num3;
	int i;
	
	for(i = 0; numDeslocamentos2e3; i++){
		if(numDeslocamentos2e3 >= (n-2-i)){
			pos_num2++;
			numDeslocamentos2e3 -= (n-2-i);
		}
		else{
			pos_num3 = pos_num2 + 1 + numDeslocamentos2e3;
			break;
		}
	}

	if(numDeslocamentos2e3 == 0){
		pos_num3 = pos_num2+1;
	}
	getSequenceLexicographically(dest+3, n-3, indexResto);
	dest[0] = (char) 1;
	
	for(i = 1; i < pos_num2; i++){
		dest[i] = dest[i+2];
	}
	dest[pos_num2] = (char) 2;


	for(i = pos_num2+1; i < pos_num3; i++){
		dest[i] = dest[i+1];
	}
	dest[pos_num3] = (char) 3;
}



//Pega a quantidade de valores menores que num na variável vet.
__device__
int qtdMenores(char* vet, int num, int n){
	int qtd = 0;
	int i;
	for(i = 0; i < n; i++){
		if(vet[i] < num)
			qtd++;
	}
	return qtd;
}

__device__
unsigned long long getIndexLexicographically(char* vet, int n){
	unsigned long long index = 0;
	int i;
	for(i = 0; i < n-1; i++){
		index += qtdMenores(vet+i+1, vet[i], n-i-1)*tex1Dfetch(fatorial,n-i-1);
	}
	return index;
}

__device__
unsigned long long getIndex(char* vet, int n){
	unsigned long long index = 0;
	int i;
	int pos_num2 = -1, pos_num3 = -1;

	//calcula os valores dos index considerando somente o 2 e 3
	for(i = 1; i < n; i++){
		if(pos_num2 == -1){
			if(vet[i] == 2){
				pos_num2 = i;
			}
			else{
				index += (n-i-1)*tex1Dfetch(fatorial,n-3);
			}
		}
		else{
			if(vet[i] == 3){
				pos_num3 = i;
				break;
			}
			else{
				index += tex1Dfetch(fatorial,n-3);
			}
		}
	}
	//calcula o valor dos index considerando os outros valores de N
	int pos = 0;
	for(i = 1; i < n; i++){
		if(i != pos_num2 && i != pos_num3){
			vet[pos] = vet[i];
			pos++;
		}
	}

	index += getIndexLexicographically(vet, n-3);

	return index;
}