import cv2 as cv
import numpy as np
from math import floor

def carregar(): 
    #Abrir imagem
    imagem = cv.imread("baixados.jpeg")
    imagem = cv.cvtColor(imagem, cv.COLOR_RGB2GRAY)
    altura, largura = imagem.shape

    #Redimensionar
    n_largura, n_altura = (int(largura/2), int(altura/2))
    imagem = cv.resize(imagem, (0,0), fx=0.469, fy=0.469)

    return imagem

def quantizar(entrada, k):
    #recriar o processo de quantização do 0
    altura, largura = entrada.shape

    #Tudo em padrões de escalas de cinza. -> Essa eu fiz na mão...
    quantizada = entrada.copy()
    valor = round(255/(k))
    for i in range(0, altura): 
        for j in range(0, largura): 
            quanta = floor(entrada[i, j]/valor)*valor
            quantizada[i, j] = quanta

    return quantizada

def apresentar_cv2(entrada, k):
    cv.imshow(f"Quantizado {k}", entrada)
    cv.waitKey(0)

def exibir_juntas(imagem, quantizadores):
    superior = quantizar(imagem, quantizadores[0])
    inferior = quantizar(imagem, quantizadores[floor(len(quantizadores)/2)])
    for i in range (1, len(quantizadores)):
        if i < len(quantizadores)/2:
            superior = np.concatenate((superior, quantizar(imagem, quantizadores[i])), axis=1)
        elif i > len(quantizadores)/2:
            inferior = np.concatenate((inferior, quantizar(imagem, quantizadores[i])), axis=1)

    superior = np.concatenate((superior , inferior), axis = 0)

    apresentar_cv2(superior, "Total")
    cv.destroyAllWindows()

def exibir_separadamente(imagem, quantizadores):
    for i in quantizadores: 
        exibir = quantizar(imagem, i)
        apresentar_cv2(exibir, i)
    cv.destroyAllWindows()

def main():
    imagem = carregar()
    quantizadores = [128, 64, 32, 16, 8, 2]
    altura, largura = imagem.shape

    #exibir individualnmente: 
    exibir_separadamente(imagem, quantizadores)

    #Exige ajustar a imagem às dimensões de tela.
    exibir_juntas(imagem, quantizadores)
    

if __name__ == "__main__": 
    main()