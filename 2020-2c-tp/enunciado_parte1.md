# 2020-2C TP
## FiuMark el cine inteligente

### Introducción:
Una famosa empresa de cine "FiuMark" quiere tener una forma automatica e inteligente de
poder predecir el comportamiento de los clientes.
Para ello desea entender la informacion que estuvo recolectando durante los ultimos 2 meses
sobre los clientes que fueron a ver la pelicula frozen 3.


Esta encuesta presentada a cada usuario que fue a ver la pelicula tenia una serie de preguntas
personales y finalmente le preguntaba si en un futuro salia frozen 4, si la volvería a ver
en ese cine o no.


La empresa quiere utilizar la informacion recolectada para dirigir campañas de marquetinkg digital
hacia los usuarios que potencialemte puedan llegar a ir a ver frozen 4.

### Tarea:

La primer tarea que nos piden es que presentemos un informe explicando que conclusiones se pueden
llegar a partir de analizar los datos recolectados. La empresa quiere saber como entender cada
uno de los datos, como se relacionan entre si y si se puede sacar alguna conclusion o descubrir
un patrón a partir de estos. Adicionalmente quieren saber cuales son los factores más importantes
que determinan si un usuario va a ir al cine a ver frozen 4 o no.


Si bien la empresa quiere empezar a usar tecnicas avanzadas de prediccion e inteligencia artificial,
todavia tiene desconfianza en las mismas (personalmente comentaron que tienen miedo que las maquinas
se revelen contra ellos) por lo que inicialmente no quieren nada complicado sino una serie muy simples
de desiciones logicas que les permitan en poco tiempo hacer una primera ronda de campaña digital. Se
espera que el este codigo simple (baseline) tenga un accuracy aceptable y que esté basada y justificada
en la investigación previa.


### Entrega:
El formato de entrega va a ser un notebook que contenga el analisis de los datos, las conclusiones a
las que se llegan a partir de ese analisis y finalmente un algoritmo simple (baseline) que intente
predecir el tarjet (si el usuario va a ver frozen 4 en el mismo cine o no).


Un comentario adicional sobre el entregable es que la empresa quiere (e hizo mucho enfasis en eso)
que el notebook esté ordenado de forma que cuente una historia, es decir, que contenga texto e imagenes que
explique cual fue el proceso de analisis que se fue haciendo, en ese sentido nos dejaron una listita
de preguntas que quieren que sigamos cuando estemos trabajando:
1. Cuales fueron las preguntas que se plantearon inicialmente?
2. Qué se hizo para responder a esas preguntas?
3. De los graficos y analisis hechos, que conclusiones se pueden sacar?
4. A partir del trabajo en los anteriores puntos, surgieron nuevas dudas? -> volver al paso 2
5. A partir de todo el analisis anterior, construir el codigo baseline que se va a usar para la
primer ronda de campaña digital, fundamentar el codigo basandose en las conclusiones de los
anteriores puntos.

### Explicación de los datos recolectados:
- **volvera**: (variable tarjet) entero que representa 0: no volvería, 1: si volvería.
- **tipo_de_salga**: El tipo de la sala (2d,3d,4d) [2d: sala comun, 3d: sala 3D, 4d: sala 4D]
- **genero**: genero el cual el usuario se identifica en la encuesta
- **edad**: Edad del usuario que completa la encuesta
- **amigos**: cantidad de amigos con los que fue a ver la pelicula (frozen 3)
- **parientes**: cantidad de familiares con los que fue a ver la pelicula (frozen 3)
- **ticket**: codigo del ticket
- **precio**: precio pagado por el ticket, en franjas de valor odenadas de 1 a 50.
- **fila**: fila dentro de la sala
- **cine**: nombre del cine [FiuMark tiene varias sedes]
- **nombre**: nombre del usuario que completa la encuesta.

Nota: algunos campos en la encuesta **pueden estar vacios**.


### Fecha de Entrega
- Entrega del notebook: Miercoles 28 de octubre.
- Defensa oral del tp: Martes 3 de noviembre.

### Condiciones de una entrega valida:
- El notebook debe poder ser ejecutado de forma secuencial de principio a fin por su corrector, todas las dependencias de librerias
deben estar en un requirements.txt.
- La funcion baseline debe llamarse baseline(X: pd.DataFrame) -> List[int].
la cual debe recibir un pandas dataframe producido de la lectura del archivo de testeo original y devolver una li

