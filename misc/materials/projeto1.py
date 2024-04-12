# Projeto 1 - Sistema de Gerenciamento de Gado em Tempo Real com IA e Visão Computacional
# Módulo Principal

# Importando bibliotecas necessárias
import math
import time
import cv2
from ultralytics import YOLO
from modulocvdsa import *
import pygame

# Configurando o OpenCV para captura de vídeo do arquivo "gado.mp4"
cap = cv2.VideoCapture("gado.mp4")

#Para Webcam descomente as linhas abaixo:
#cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)

# Definindo o idioma para o sistema (os nomes das classes no modelo YOLO estão em inglês)
language = "en"

# Carregando o modelo de detecção YOLO com o arquivo de pesos "yolov8n.pt"
modelo_dsa = YOLO("yolov8n.pt")

# Lista de nomes de classes para detecção pelo modelo YOLO
classNames = ['Dog', 'Koala', 'Zebra', 'pig', 'antelope', 'badger', 'bat', 'bear', 'bison', 'cat', 'chimpanzee', 
              'cow', 'coyote', 'deer', 'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish', 
              'goose', 'gorilla', 'hamster', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
              'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'pigeon',
              'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'porcupine', 
              'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
              'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'woodpecker']

# Inicializando o rastreador SORT com parâmetros específicos
# A função Sort() está no arquivo modulocvdsa.py
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

# Definindo limites para alguma funcionalidade específica no sistema
limits = [400,400,700,700]
limitT = [400,400,700,700]
totalCount = []

# Loop principal do programa
while True:

    # Lendo uma imagem do vídeo (cada frame do vídeo é tratado como uma imagem)
    success, img = cap.read()
    
    # Obtendo resultados do modelo YOLO para a imagem capturada
    resultados = modelo_dsa(img, stream = True)

    # Inicializando um array vazio para armazenar as detecções
    detections = np.empty((0,5))

    # Iterando sobre os resultados da detecção
    for r in resultados:
        
        # Extraindo as caixas delimitadoras dos resultados
        boxes = r.boxes
        
        # Iterando sobre cada caixa delimitadora
        for box in boxes:

            # Extraindo coordenadas da caixa delimitadora
            x1,y1,x2,y2 = box.xyxy[0]
            
            # Convertendo coordenadas para inteiros
            x1, y1, x2, y2 = int(x1), int(y1), int(x2),int(y2)
            
            # Desenhando a caixa delimitadora na imagem
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # Calculando a largura e altura da caixa delimitadora
            w, h = x2-x1, y2-y1
         
            # Calculando e arredondando a confiança da detecção
            conf = math.ceil((box.conf[0]*100))/100           
    
    # Atualizando o rastreador com as novas detecções
    resultadosTracker = tracker.update(detections)
        
    # Iterando sobre os resultados do rastreador
    for resultado in resultadosTracker:
        
        # Extraindo coordenadas e ID do resultado do rastreador
        x1,y1,x2,y2,id = resultado
        
        # Convertendo coordenadas para inteiros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Imprimindo o resultado do rastreador
        print(resultado)
    
    # Tentativa de exibir a imagem processada
    try:
        cv2.imshow("Image", img)
    except:
        # Se houver um erro, interrompe o loop
        break
    
    # Aguardando uma tecla ser pressionada (com um delay de 1 milissegundo)
    cv2.waitKey(1)

