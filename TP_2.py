import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ==========================================
# PROBLEMA 2 - Corrección de multiple choice
# ==========================================

"""
n = int(input("Ingrese un número del 1 al 5: "))
while n not in range(1,6):
	n = int(input("Error: ingrese un número del 1 al 5: "))
img = cv2.imread(f'multiple_choice_{n}.png',cv2.IMREAD_GRAYSCALE)
img.shape
"""


img = cv2.imread(f'multiple_choice_1.png',cv2.IMREAD_GRAYSCALE)
img.shape

# ITEM A
# Usamos la gráfica para delimitar la zona de las preguntas
#plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
#plt.show()

index_respuestas = {"y1":140, "y2":1038}
img_respuestas = img[index_respuestas["y1"]:index_respuestas["y2"]]
#plt.figure(), plt.imshow(img_respuestas, cmap='gray'), plt.show(block=False)
#plt.show()

print(f"Tamaño de imagen donde están las respuestas: {img_respuestas.shape}")


black = 0
img_zeros = img_respuestas==black
                           
img_row_zeros = img_zeros.any(axis=1)
img_row_zeros_idxs = np.argwhere(img_zeros.any(axis=1))

#plt.figure(), plt.imshow(escrito, cmap='gray'), plt.show()
#plt.show()

x = np.diff(img_row_zeros)          
renglones_indxs = np.argwhere(x)
print(f"Cantidad de renglones: {int(len(renglones_indxs)/2)}")


# Visualización
"""
ii = np.arange(0,len(renglones_indxs),2)    # 0 2 4 ... X --> X es el último nro par antes de len(renglones_indxs)
renglones_indxs[ii]+=1

xri = np.zeros(img_respuestas.shape[0])
xri[renglones_indxs] = (img_respuestas.shape[1]-1)
yri = np.arange(img_respuestas.shape[0])            
plt.figure(), plt.imshow(img_respuestas, cmap='gray'), plt.plot(xri, yri, 'r'), plt.title("Renglones - Inicio y Fin"), plt.show()
"""

# Genero estructura de datos para guardar datos de renglones
r_idxs = np.reshape(renglones_indxs, (-1,2))
renglones = []
for ir, idxs in enumerate(r_idxs):
    renglones.append({
        "ir": ir+1,
        "cord": idxs,
        "img": img_respuestas[idxs[0]:idxs[1], :]
    })

# Exploramos el dato y lo visualizamos.
#print(renglones[24])
#plt.figure(), plt.imshow(renglones[24]["img"], cmap='gray'), plt.show()


# Busquemos ahora inicio y fin de cada "columna" (número de pregunta, punto, A, B, C, D y E)

letras = []
il = -1
for ir, renglon in enumerate(renglones):
    renglon_zeros = renglon["img"]==0

    # Analizo columnas del renglón 
    ren_col_zeros = renglon_zeros.any(axis=0)
    ren_col_zeros_idxs = np.argwhere(renglon_zeros.any(axis=0))
    # Visualizo
    xc = np.arange(renglon_zeros.shape[1])
    yc = ren_col_zeros*(renglon_zeros.shape[0]-1)
    plt.figure(), plt.imshow(renglon_zeros, cmap='gray'), plt.plot(xc, yc, c='b'), plt.title(f"Renglón {ir+1}"), plt.show()        
        
    # Encontramos inicio y final de cada letra
    x = np.diff(ren_col_zeros)
    letras_indxs = np.argwhere(x) 
    # *** Modifico índices ***********
    ii = np.arange(0,len(letras_indxs),2)
    letras_indxs[ii]+=1

    letras_indxs = letras_indxs.reshape((-1,2))
    
    for irl, idxs in enumerate(letras_indxs):
        il+=1
        letras.append({
            "ir":ir+1,
            "irl":irl+1,
            "il": il,
            "cord": [renglon["cord"][0], idxs[0], renglon["cord"][1], idxs[1]],
            "img": renglon["img"][:, idxs[0]:idxs[1]]
        })



"""
# ITEM B (NOS PUSIMOS CREATIVOS)
# Exploración de imágenes


# Usamos la gráfica para delimitar el renglón con la información relevante
#plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
#plt.show()

# Indexamos el renglón, y el resto de datos en ese renglón
index_renglon = {"y1":109, "y2":129}
index_name = {"x1":98, "x2":280}
index_id = {"x1":332, "x2":430}
index_code = {"x1":432, "x2":565}
index_date = {"x1":650, "x2":767}


# Verificamos que los recortes funcionen
img_renglon = img[index_renglon["y1"]:index_renglon["y2"]]
#plt.figure(), plt.imshow(img_renglon, cmap='gray'), plt.show(block=False)
#plt.show()

img_name = img[index_renglon["y1"]:index_renglon["y2"],index_name["x1"]:index_name["x2"]]
img_id = img[index_renglon["y1"]:index_renglon["y2"],index_id["x1"]:index_id["x2"]]
# A code le dejamos el título, sino la información de la imagen era muy poca y no se podía leer.
img_code = img[index_renglon["y1"]:index_renglon["y2"],index_code["x1"]:index_code["x2"]]
img_date = img[index_renglon["y1"]:index_renglon["y2"],index_date["x1"]:index_date["x2"]]

img_datos = [img_name,img_id,img_code,img_date]
for i in range(4):
	#plt.subplot(1, 4, i+1)
	#plt.imshow(img_datos[i], cmap='gray')
    pass
#plt.show()

# Usaremos PyTesseract y Poppler (adaptación de C++) como OCR (optical characters recognition)
# para ayudarnos a leer las imágenes.


# Directorio para guardar imagenes auxiliares
carpeta = 'images'

# Crear la carpeta si no existe
import os
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Guardar la imagen y el dato que allí leemos con el procesador óptico
ruta_imagen = os.path.join(carpeta, 'name.png')
plt.imsave(ruta_imagen, img_name, cmap='gray')
imagen = Image.open(ruta_imagen)
name = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()

ruta_imagen = os.path.join(carpeta, 'id.png')
plt.imsave(ruta_imagen, img_id, cmap='gray')
imagen = Image.open(ruta_imagen)
id = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()
# A veces aparecían comas al principio o al final de id
id = id.lstrip(",")
id = id.rstrip(",")

ruta_imagen = os.path.join(carpeta, 'code.png')
plt.imsave(ruta_imagen, img_code, cmap='gray')
imagen = Image.open(ruta_imagen)
code = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()
code = code[-1]

ruta_imagen = os.path.join(carpeta, 'date.png')
plt.imsave(ruta_imagen, img_date, cmap='gray')
imagen = Image.open(ruta_imagen)
date = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()



if len(name.split())==2 and len(name)<=25:
    print(f"Nombre ({name}): OK")
else:
    print(f"Nombre ({name}): MAL")
if len(id.split())==1 and len(id)==8:
    print(f"Id ({id}): OK")
else:
    print(f"Id ({id}): MAL")
if len(code)==1:
    print(f"Code ({code}): OK")
else:
    print(f"Code ({code}): MAL")
if len(date.split())==1 and len(date)==8:
    print(f"Date ({date}): OK")
else:
    print(f"Date ({date}): MAL")

	"""