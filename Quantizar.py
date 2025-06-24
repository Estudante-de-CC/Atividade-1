import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar(): 
    #Testei com 3 imagens diferentes também. Funcionou. 
    imagem = cv2.imread("cores.jpeg")
    altura, largura, nhe = imagem.shape
    #Redimensionar para o tamanho da tela
    imagem = cv2.resize(imagem, (int(largura/2), int(altura/2)))
    #imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    return imagem

def quantizar(entrada, k):

    #Essa função está bem parecida com o presente da demonstração. Eu usei a de lá como exemplo pra essa parte
    #Estava tentando entender como a função cv2.kmeans funcionava... 
    D = entrada.reshape((-1,3))
    
    D = np.float32(D)
    
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    K = k
    ret,label,centro=cv2.kmeans(D,K,None,criterios,2,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(centro)
    res = center[label.flatten()]
    res2 = res.reshape((entrada.shape))

    return res2

def apresentar_cv2(entrada, k):
    cv2.namedWindow(f"VASH! Quantizado {k}")
    cv2.imshow(f"VASH! Quantizado {k}", entrada)
    cv2.waitKey(0)

def main():
    imagem = carregar()
    quantizadores = [128, 64, 32, 16, 8, 2]
    j = 0  
    for i in quantizadores:
        exibir = quantizar(imagem, i)
        apresentar_cv2(exibir, i)
    
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()

