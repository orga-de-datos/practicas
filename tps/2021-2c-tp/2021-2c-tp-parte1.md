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
Tener en cuenta que pueden tener dos notebooks, uno con lo mas importante para presentar durante la reunion de 20 min, 
y otro con graficos y analisis adicionales que no son importantes para la reunión.   

### Explicación de los datos recolectados:

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
**velocidad_viendo_tarde**: velocidad del viento medido en km/h (medicion tomada por Flint a la tardecita)  
**humedad_temprano**: porcentaje de humedad (medicion tomada cuando Flint se levanta a la mañana)  
**humedad_tarde**: porcentaje de humedad (medicion tomada por Flint a la tardecita)  
**presion_atmosferica_temprano**: presion atmosférica (hectopascales) (medicion tomada cuando Flint se levanta a la mañana)  
**presion_atmosferica_tarde**: presion atmosférica (hectopascales) (medicion tomada por Flint a la tardecita)  
**nubosidad_temprano**: nubosidad en el cielo (de 0 a 8) (medicion tomada cuando Flint se levanta a la mañana)  
**nubosidad_tarde**: nubosidad en el cielo (de 0 a 8) (medicion tomada por Flint a la tardecita)  
**temperatura_temprano**: temperatura (en grados centigrados) (medicion tomada cuando Flint se levanta a la mañana)  
**temperatura_tarde**: temperatura (en grados centigrados) (medicion tomada por Flint a la tardecita)  
**llovieron_hamburguesas_hoy**: booleano registrando si cayeron hamburguesas ese día  
**llovieron_hamburguesas_al_dia_siguiente**: booleano registrando si cayeron hamburguesas el dia siguiente (variable tarjet)  


Nota1: algunos campos en la encuesta **pueden estar vacíos**.   

Los datos se encuentra en: 
- https://docs.google.com/spreadsheets/d/1gvZ03uAL6THwd04Y98GtIj6SeAHiKyQY5UisuuyFSUs
- https://docs.google.com/spreadsheets/d/1wduqo5WyYmCpaGnE81sLNGU0VSodIekMfpmEwU0fGqs


### Fecha de Entrega
- Entrega del notebook: Miercoles 6 de Octubre. En un repo de github PRIVADO (el que les fue asignado desde el curso)
- Defensa oral del TP: Martes 12 de Octubre.

### Condiciones de una entrega válida:
- El notebook debe poder ser ejecutado de forma secuencial de principio a fin por su corrector, todas las dependencias 
  de librerías deben estar en un requirements.txt.
- La función baseline debe llamarse baseline(X: pd.DataFrame) -> List[int].
la cual debe recibir un pandas dataframe producido de la lectura del archivo de testeo original y devolver una lista
  con las predicciones (1 para si es que van a llover hamburguesas mañana, 0 si no van a llover hamburguesas mañana)

