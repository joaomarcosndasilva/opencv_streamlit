import cv2
import streamlit as st
from PIL import Image
import numpy as np
from skimage import morphology, io, color, feature, filters

def brilho_imagem(imagem, resultado):
	img_brilho = cv2.convertScaleAbs(imagem, beta=resultado)
	return img_brilho

def borra_imagem(imagem, resultado):
	img_borrada = cv2.GaussianBlur(imagem, (7,7), resultado)
	return img_borrada

def melhora_detalhe(imagem):
	img_melhorada = cv2.detailEnhance(imagem, sigma_s=34, sigma_r=0.50)
	return img_melhorada

def escala_cinza(imagem):
	img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	return img_cinza



def principal():
	st.title('OpenCV Data App')
	st.subheader('Esse aplicativo permite integrar processamento de imagens com OPenCV')
#######################################################################
	#Para adicionar o arquivo de upload:
	arquivo_imagem = st.file_uploader('Envie Sua Imagem e role para baixo para ver o resultado', type=['jpg', 'png', 'jpeg'] )
	st.sidebar.header('Controles de formatação')
############################### Criando os controles para manutenção ###########
    #st.sidebar.header('Controles para mudanças na imagem')
	taxa_borrao = st.sidebar.slider('Borrão', min_value=0.2, max_value=3.5)
	qtd_brilho = st.sidebar.slider('Brilho', min_value=-50, max_value=50, value=0)
	filtro = st.sidebar.checkbox('Melhorar Detalhes da Imagem')
	img_cinza = st.sidebar.checkbox('Imagem em escala de cinza')
	img_erosao = st.sidebar.checkbox('Filtro Erosão')
	img_dilatacao = st.sidebar.checkbox('Filtro Dilatação')
	img_edge = st.sidebar.checkbox('Filtro Edge')


	if not arquivo_imagem:
		return None
	st.text('>>Role para baixo e ajuste os filtros em tempo real<<')

	imagem_original = Image.open(arquivo_imagem)
	imagem_original = np.array(imagem_original)

	imagem_processada = borra_imagem(imagem_original, taxa_borrao)

	imagem_processada = brilho_imagem(imagem_processada, qtd_brilho)

	if filtro:
		imagem_processada = melhora_detalhe(imagem_processada)

	if img_cinza:
		imagem_processada = escala_cinza(imagem_processada)

	if img_erosao:
		imagem_processada = morphology.erosion(imagem_processada)

	if img_dilatacao:
		imagem_processada = morphology.dilation(imagem_processada)

	if img_edge:
		imagem_processada = filters.sobel(imagem_processada)

#####################################################
	img = cv2.imread('DKC.jpg')

	st.text('Imagem Original x Imagem Processada')

	st.image([imagem_original, imagem_processada])


if __name__ == '__main__':
	principal()

