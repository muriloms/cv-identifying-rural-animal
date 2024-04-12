# Projeto 1 - Sistema de Gerenciamento de Gado em Tempo Real com IA e Visão Computacional
# Módulo de Tratamento dos Frames do Vídeo


# Imports
import os
import glob
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from filterpy.kalman import KalmanFilter


# Random seed
np.random.seed(0)


# Esta função abaixo implementa uma função para realizar o alinhamento linear (ou atribuição de custo mínimo) 
# que é frequentemente usado em algoritmos de rastreamento, como o Deep Sort Algorithm. A função primeiro tenta usar 
# a biblioteca lap para uma solução mais eficiente e, se não estiver disponível, recorre ao linear_sum_assignment do SciPy.

# Definição da função para alinhamento linear usando Deep Sort Algorithm (DSA)
def dsa_alinhamento_linear(cost_matrix):
  
  # Tentativa de executar o algoritmo
  try:
    
    # Importando a biblioteca lap para cálculo de alinhamento linear
    import lap
    
    # Utilizando o algoritmo LAPJV para resolver o problema de atribuição linear e estendendo o custo
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    
    # Retornando um array numpy com os índices de alinhamento (apenas para pares válidos)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  
  # Capturando o erro se a biblioteca lap não estiver instalada
  except ImportError:

    # Importando a função de atribuição de soma linear do scipy
    from scipy.optimize import linear_sum_assignment
        
    # Realizando a atribuição de soma linear usando o scipy
    x, y = linear_sum_assignment(cost_matrix)
    
    # Retornando um array numpy com os índices de alinhamento
    return np.array(list(zip(x, y)))


# Esta função abaixo calcula a Interseção sobre União (IoU) entre duas caixas delimitadoras, uma técnica comum 
# em Visão Computacional para avaliar a precisão de algoritmos de detecção de objetos. A função usa operações 
# de broadcasting do NumPy para processar eficientemente vários pares de caixas delimitadoras.

# Define a função para calcular a Interseção sobre União (IoU) em lote entre caixas delimitadoras
def dsa_iou_batch(bb_test, bb_gt):

  # Expande a dimensão do ground truth (bb_gt) para possibilitar operações de broadcasting
  bb_gt = np.expand_dims(bb_gt, 0)
  
  # Expande a dimensão do teste (bb_test) para possibilitar operações de broadcasting
  bb_test = np.expand_dims(bb_test, 1)
  
  # Calcula o máximo dos x1 (canto superior esquerdo) para interseção
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  
  # Calcula o máximo dos y1 (canto superior esquerdo) para interseção
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  
  # Calcula o mínimo dos x2 (canto inferior direito) para interseção
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  
  # Calcula o mínimo dos y2 (canto inferior direito) para interseção
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  
  # Calcula a largura da área de interseção
  w = np.maximum(0., xx2 - xx1)
  
  # Calcula a altura da área de interseção
  h = np.maximum(0., yy2 - yy1)
  
  # Calcula a área da interseção
  wh = w * h
  
  # Calcula a IoU (Interseção sobre União) para cada par de caixas
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  
  # Retorna a IoU calculada
  return(o)  


# Esta função abaixo é usada para converter uma caixa delimitadora, que normalmente é dada pelas coordenadas 
# do canto superior esquerdo e do canto inferior direito, para um vetor de estado que inclui a posição central (x, y), 
# a escala (área da caixa) e a razão de aspecto. Este formato é útil em algoritmos de rastreamento como parte 
# do processo de filtragem e previsão de estado.

# Define uma função para converter uma caixa delimitadora para o formato do vetor de estado z
def dsa_convert_bbox_to_z(bbox):

  # Calcula a largura da caixa delimitadora
  w = bbox[2] - bbox[0]
  
  # Calcula a altura da caixa delimitadora
  h = bbox[3] - bbox[1]
  
  # Calcula a coordenada x do centro da caixa delimitadora
  x = bbox[0] + w/2.
  
  # Calcula a coordenada y do centro da caixa delimitadora
  y = bbox[1] + h/2.
  
  # Calcula a escala (área) da caixa delimitadora
  s = w * h    # A escala é simplesmente a área
  
  # Calcula a razão de aspecto da caixa delimitadora
  r = w / float(h)
  
  # Retorna o vetor de estado z no formato [x, y, s, r], como um array numpy de forma (4,1)
  return np.array([x, y, s, r]).reshape((4, 1))


# Esta função abaixo converte um vetor de estado no formato centro-forma [x, y, s, r] (onde x, y é o centro, 
# s é a escala/área e r é a razão de aspecto) de volta para uma caixa delimitadora no formato tradicional [x1, y1, x2, y2]. 
# Ela também suporta a inclusão de uma pontuação (score) se fornecida, que é útil para aplicações de rastreamento 
# e detecção de objetos.

# Define uma função para converter um vetor de estado para uma caixa delimitadora
def dsa_convert_x_to_bbox(x, score=None):

  # Calcula a largura da caixa delimitadora usando a escala e a razão de aspecto
  w = np.sqrt(x[2] * x[3])
  
  # Calcula a altura da caixa delimitadora
  h = x[2] / w

  # Verifica se uma pontuação foi fornecida
  if(score==None):
    # Retorna a caixa delimitadora no formato [x1,y1,x2,y2] sem pontuação
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    # Retorna a caixa delimitadora no formato [x1,y1,x2,y2] com pontuação
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


# Esta função abaixo é parte essencial de sistemas de rastreamento de objetos, como o Deep Sort, 
# onde as detecções (objetos detectados em um frame atual) são associadas a rastreadores (objetos rastreados ao longo do tempo) 
# usando a métrica de sobreposição IoU (Interseção sobre União).

# Define uma função para associar detecções a rastreadores usando o limiar de IoU
def dsa_associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):

  # Verifica se a lista de rastreadores está vazia e retorna arrays vazios caso esteja
  if(len(trackers)==0):
    return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

  # Calcula a matriz de IoU entre as detecções e os rastreadores
  iou_matrix = dsa_iou_batch(detections, trackers)

  # Verifica se a matriz de IoU não está vazia
  if min(iou_matrix.shape) > 0:
    
    # Cria uma matriz binária indicando se a IoU excede o limiar
    a = (iou_matrix > iou_threshold).astype(np.int32)
    
    # Se cada detecção e rastreador só tem um correspondente, usa simples associação
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      # Caso contrário, usa alinhamento linear para associação
      matched_indices = dsa_alinhamento_linear(-iou_matrix)
  else:
    # Se a matriz de IoU está vazia, não há correspondências
    matched_indices = np.empty(shape=(0,2))

  # Inicializa a lista de detecções não correspondidas
  unmatched_detections = []
  
  # Percorre todas as detecções para encontrar as não correspondidas
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  
  # Inicializa a lista de rastreadores não correspondidos
  unmatched_trackers = []
  
  # Percorre todos os rastreadores para encontrar os não correspondidos
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filtra correspondências com baixa IoU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]] < iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  
  # Se não houver correspondências, retorna um array vazio
  if(len(matches) == 0):
    matches = np.empty((0,2), dtype=int)
  else:
    # Caso contrário, concatena as correspondências
    matches = np.concatenate(matches, axis=0)

  # Retorna as correspondências, detecções não correspondidas e rastreadores não correspondidos
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# Esta classe abaixo implementa um rastreador de caixas delimitadoras usando um filtro de Kalman, 
# comum em sistemas de rastreamento de objetos. O filtro de Kalman é configurado para um modelo de velocidade constante. 
# A classe gerencia o estado interno do objeto rastreado, incluindo previsões e atualizações com base nas observações recebidas. 
# Ela também mantém um histórico das estimativas de posição, além de gerenciar seu próprio identificador único e outros 
# metadados relacionados ao rastreamento.

# O filtro de Kalman ajuda a prever a posição e a velocidade de um objeto em movimento de frame para frame, 
# mesmo quando o objeto está temporariamente fora da visão.

# Definição da classe KalmanBoxTracker, que representa o estado interno de objetos rastreados observados como caixas delimitadoras
class KalmanBoxTracker(object):

  # Contador estático para rastrear o número de instâncias da classe
  count = 0

  # Método construtor da classe
  def __init__(self,bbox):

    # Inicializa o filtro de Kalman com 7 dimensões para o estado e 4 para as medições
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 

    # Define a matriz de transição de estado (F) para um modelo de velocidade constante
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    
    # Define a matriz de observação (H)
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

     # Ajusta a matriz de covariância do ruído de observação (R)
    self.kf.R[2:,2:] *= 10.

    # Ajusta a matriz de covariância do erro do estado inicial (P)
    self.kf.P[4:,4:] *= 1000.
    self.kf.P *= 10.

    # Ajusta a matriz de covariância do ruído do processo (Q)
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    # Converte a caixa delimitadora inicial para o formato do vetor de estado e o atribui ao filtro
    self.kf.x[:4] = dsa_convert_bbox_to_z(bbox)

    # Inicializa o tempo desde a última atualização
    self.time_since_update = 0

    # Inicializa o tempo desde a última atualização
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1

    # Inicializa o histórico de previsões
    self.history = []

    # Inicializa o contador de detecções (hits)
    self.hits = 0

    # Inicializa a sequência de acertos (hit streak)
    self.hit_streak = 0

    # Inicializa a idade do rastreador
    self.age = 0

  # Método para atualizar o estado do rastreador com uma nova caixa delimitadora observada
  def update(self,bbox):

    # Reseta o tempo desde a última atualização
    self.time_since_update = 0

    # Limpa o histórico de previsões
    self.history = []

    # Incrementa o contador de detecções e a sequência de acertos
    self.hits += 1
    self.hit_streak += 1

    # Atualiza o estado do filtro de Kalman com a nova caixa delimitadora
    self.kf.update(dsa_convert_bbox_to_z(bbox))

  # Método para prever o próximo estado (posição da caixa delimitadora)
  def predict(self):
  
    # Ajusta a velocidade se a soma da velocidade e da escala for menor ou igual a zero
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0

    # Realiza a previsão do próximo estado
    self.kf.predict()

    # Incrementa a idade do rastreador
    self.age += 1

    # Se o tempo desde a última atualização for maior que zero, reseta a sequência de acertos
    if(self.time_since_update>0):
      self.hit_streak = 0

    # Incrementa o tempo desde a última atualização
    self.time_since_update += 1

    # Adiciona a previsão atual ao histórico
    self.history.append(dsa_convert_x_to_bbox(self.kf.x))

    # Retorna a última previsão do histórico
    return self.history[-1]

  # Método para obter o estado atual (estimativa da caixa delimitadora)
  def get_state(self):

    # Retorna a caixa delimitadora atual convertida a partir do estado do filtro de Kalman
    return dsa_convert_x_to_bbox(self.kf.x)


# Este código define a classe `Sort`, que implementa o algoritmo Simple Online and Realtime Tracking (SORT). 
# O algoritmo utiliza o filtro de Kalman para prever a posição dos objetos e associa as detecções aos 
# rastreadores existentes usando a métrica de Interseção sobre União (IoU). Os rastreadores são atualizados 
# com cada frame e novos rastreadores são criados para detecções que não correspondem aos existentes. 
# Os rastreadores antigos são removidos se não forem atualizados por um período especificado (`max_age`). 
# A classe retorna as estimativas atuais de localização dos objetos rastreados.

# # Definição da classe Sort, que implementa o algoritmo Simple Online and Realtime Tracking
class Sort(object):

  # Método construtor da classe
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):

    # Inicializa a idade máxima de um rastreador sem atualizações
    self.max_age = max_age

    # Inicializa o número mínimo de hits (deteções) para considerar um rastreador válido
    self.min_hits = min_hits

    # Inicializa o limiar de IoU para associar detecções a rastreadores
    self.iou_threshold = iou_threshold

    # Lista para armazenar os rastreadores ativos
    self.trackers = []

    # Contador para acompanhar o número de frames processados
    self.frame_count = 0

  # Método para atualizar os rastreadores com novas detecções
  def update(self, dets=np.empty((0, 5))):

    # Incrementa o contador de frames
    self.frame_count += 1

    # Inicializa um array para armazenar os estados dos rastreadores
    trks = np.zeros((len(self.trackers), 5))

    # Lista para armazenar índices dos rastreadores a serem removidos
    to_del = []

    # Lista para armazenar os resultados
    ret = []

    # Itera sobre os rastreadores para atualizar e limpar
    for t, trk in enumerate(trks):

      # Obtém a previsão atual do rastreador
      pos = self.trackers[t].predict()[0]

      # Atualiza o array de rastreadores com a previsão
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

      # Se houver algum valor NaN, marca o rastreador para exclusão
      if np.any(np.isnan(pos)):
        to_del.append(t)

    # Comprime as linhas do array, excluindo valores inválidos
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

    # Remove os rastreadores marcados para exclusão
    for t in reversed(to_del):
      self.trackers.pop(t)

    # Associa as detecções aos rastreadores existentes
    matched, unmatched_dets, unmatched_trks = dsa_associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # Atualiza os rastreadores com as detecções correspondentes
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # Cria novos rastreadores para detecções não correspondidas
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)

    # Inicializa um contador para os rastreadores
    i = len(self.trackers)

    # Itera sobre os rastreadores para gerar saídas e remover os antigos
    for trk in reversed(self.trackers):

        # Obtém o estado atual do rastreador
        d = trk.get_state()[0]

        # Verifica se o rastreador deve ser incluído na saída
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):

          # Adiciona o estado do rastreador à lista de resultados
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
          
        i -= 1

         # Remove rastreadores antigos com base no tempo de atualização
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)

    # Se houver resultados, os concatena e retorna
    if(len(ret)>0):
      return np.concatenate(ret)

    # Se não houver resultados, retorna um array vazio
    return np.empty((0,5))


