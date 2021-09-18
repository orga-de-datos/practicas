# 2021-2C TP
## FIUFIP: Lluvia de Hamburgesas

### Introducción:
Flint Lockwood es un científico loco que ha estado trabajando en una máquina que hace llover hamburguesas, 
lamentablemente la máquina se ha vuelto loca y Flint no puede controlar cuando la máquina se activa
produciendo asi que lluevan hamburguesas.

Pero él sospecha que la máquina tiene un patron por lo cual ha estado recolectando datos del comportamiento
de la máquina y otras variables. Nos ha pedido ayuda para lograr entender esos datos (él está ocupado 
tratando de encontrar la manera de desactivar la máquina)

Estos datos consisten en una serie de atributos climáticos por dia y si la máquina se activó al dia siguiente o no.

Flint quiere utilizar la información recolectada para poder predecir si al dia siguiente la máquina se va a activar o no
y poder preparar medidas para contrarrestar a las hamburguesas del cielo.

### Tarea:

La primer tarea que nos piden es que presentemos un informe explicando qué conclusiones se pueden
llegar a partir de analizar los datos recolectados. Flint quiere saber cómo entender cada
uno de los datos, cómo se relacionan entre sí y si se puede sacar alguna conclusión o descubrir
un patrón a partir de estos. Adicionalmente quieren saber cuáles son los factores más importantes
que determinan si la máquina se va a activar al dia siguiente o no.


Si bien Flint quiere empezar a usar técnicas avanzadas de predicción e inteligencia artificial,
todavía tiene desconfianza en las mismas (dada la experiencia traumatica que esta teniendo en este momento) 
por lo que inicialmente no quieren nada complicado sino una serie muy simple
de decisiones lógicas que les permitan en poco tiempo hacer una primera ronda de predicciones. Se
espera que este código simple (baseline) tenga una accuracy aceptable (mayor a ??%) y que esté basada y justificada
en la investigación previa.


### Entrega:
El formato de entrega va a ser un notebook que contenga el análisis de los datos, las conclusiones a
las que se llegan a partir de ese análisis y finalmente un algoritmo simple (baseline) que intente
predecir el target (si la máquina se va a activar al dia siguiente).


Un comentario adicional sobre el entregable es que la empresa quiere (e hizo mucho énfasis en eso)
que el notebook esté ordenado de forma que cuente una historia, es decir, que contenga texto e imágenes que
expliquen cuál fue el proceso de análisis que se fue haciendo. En ese sentido, nos dejaron una listita
de preguntas que quieren que sigamos cuando estemos trabajando:
1. ¿Cuáles fueron las preguntas que se plantearon inicialmente?
2. ¿Qué se hizo para responder a esas preguntas?
3. De los gráficos y análisis hechos, ¿qué conclusiones se pueden sacar?
4. A partir del trabajo en los anteriores puntos, ¿surgieron nuevas dudas? -> Volver al paso 2
5. A partir de todo el análisis anterior, construir el código baseline que se va a usar para la
primera ronda de campaña digital. Fundamentar el código basándose en las conclusiones de los
anteriores puntos.
   
Se espera que presenten el notebook en una charla de 20 minutos y dijo que quiere explícitamente el formato
(por cada pregunta):
- pregunta
- gráfico/s para responder esa pregunta
- por cada gráfico quieren un comentario escrito de que se interpreta en el mismo
- respuesta a la pregunta en base a todos los gráficos y análisis de los mismos  

Sin este formato, el tp no va a ser aceptado.

### Explicación de los datos recolectados:

WIP WIP WIP - START

"Date": "dia"
"Location": "barrio" (remap a barrios - asignando los barrios de acuerdo a cant de gente)
"MinTemp": "temp_min'="
"MaxTemp": "temp_max"
"Rainfall": "mm_lluvia_dia"
"Evaporation": "mm_evaporados_agua"
"Sunshine": "horas_de_sol"
"WindGustDir": "rafaga_viento_max_direccion"
"WindGustSpeed": "rafaga_viento_max_velocidad"
"WindDir9am": "direccion_viento_temprano"
"WindDir3pm": "direccion_viento_tarde"
"WindSpeed9am": "velocidad_viendo_temprano"
"WindSpeed3pm": "velocidad_viendo_tarde"
"Humidity9am": "humedad_temprano'
"Humidity3pm": "humedad_tarde"
"Pressure9am": "presion_atmosferica_temprano"
"Pressure3pm": "presion_atmosferica_tarde"
"Cloud9am" : "nubosidad_temprano"
"Cloud3pm": "nubosidad_tarde"
"Temp9am": "temperatura_temprano"
"Temp3pm": "temperatura_tarde"
"RainToday": "llovieron_hamburguesas_hoy"
"RainTomorrow": "llovieron_hamburguesas_al_dia_siguiente"

**dia**: dia en el cual se tomó la medición  
**barrio**: barrio donde se tomó la medición  
**temp_min**: temperatura maxima registrada en el día  
**temp_max**: temperatura minima registrada en el día  
**mm_lluvia_dia**: milimetros que llovieron en el día  
**mm_evaporados_agua**: milimetros que se evaporaron en el día  
**horas_de_sol**: horas totales de sol en el día  
**rafaga_viento_max_direccion**: dirección de la ráfaga de viento mas fuerte detectada en el día
**rafaga_viento_max_velocidad**: velocidad medida en km/h de la ráfaga de viento más fuerte detectada en el día
**direccion_viento_temprano**: dirección del viento (medicion tomada cuando Flint se levanta a la mañana)
**direccion_viento_tarde**: dirección del viento (medicion tomada por Flint a la tardecita)
**velocidad_viendo_temprano**: velocidad del viento medido en km/h (medicion tomada cuando Flint se levanta a la mañana)
**velocidad_viendo_tarde**
**humedad_temprano**
**humedad_tarde**
**presion_atmosferica_temprano**
**presion_atmosferica_tarde**
**nubosidad_temprano**
**nubosidad_tarde**
**temperatura_temprano**
**temperatura_tarde**
**llovieron_hamburguesas_hoy**
**llovieron_hamburguesas_al_dia_siguiente**

WIP WIP WIP - END

Nota1: algunos campos en la encuesta **pueden estar vacíos**.  

El link de los datos se encuentra en: https://docs.google.com/spreadsheets/d/<TODO>


### Fecha de Entrega
- Entrega del notebook: Miercoles 6 de Octubre. En un repo de github PRIVADO (el que les fue asignado desde el curso)
- Defensa oral del TP: Martes 12 de Octubre.

### Condiciones de una entrega válida:
- El notebook debe poder ser ejecutado de forma secuencial de principio a fin por su corrector, todas las dependencias 
  de librerías deben estar en un requirements.txt.
- La función baseline debe llamarse baseline(X: pd.DataFrame) -> List[int].
la cual debe recibir un pandas dataframe producido de la lectura del archivo de testeo original y devolver una lista
  con las predicciones (1 para si es que van a llover hamburguesas mañana, 0 si no van a llover hamburguesas mañana)

