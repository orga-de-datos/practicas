# 2020-2C TP
## FiuMark: El Cine Inteligente

### Introducción:
Una famosa empresa de cine "FiuMark" quiere tener una forma automática e inteligente de
poder predecir el comportamiento de los clientes.
Para ello desea entender la información que estuvo recolectando durante los últimos 2 meses
sobre los clientes que fueron a ver la película Frozen 3.


Esta encuesta presentada a cada usuario que fue a ver la película tenia una serie de preguntas
personales y finalmente le preguntaba que si en un futuro salía Frozen 4, la volvería a ver
en ese cine o no.


La empresa quiere utilizar la información recolectada para dirigir campañas de marketing digital
hacia los usuarios que potencialmente pueden llegar a ir a ver Frozen 4.

### Tarea:

La primer tarea que nos piden es que presentemos un informe explicando qué conclusiones se pueden
llegar a partir de analizar los datos recolectados. La empresa quiere saber cómo entender cada
uno de los datos, cómo se relacionan entre sí y si se puede sacar alguna conclusión o descubrir
un patrón a partir de estos. Adicionalmente quieren saber cuáles son los factores más importantes
que determinan si un usuario va a ir al cine a ver Frozen 4 o no.


Si bien la empresa quiere empezar a usar técnicas avanzadas de predicción e inteligencia artificial,
todavía tiene desconfianza en las mismas (personalmente comentaron que tienen miedo que las máquinas
se revelen contra ellos) por lo que inicialmente no quieren nada complicado sino una serie muy simple
de decisiones lógicas que les permitan en poco tiempo hacer una primera ronda de campaña digital. Se
espera que este código simple (baseline) tenga una accuracy aceptable (mayor a 65%) y que esté basada y justificada
en la investigación previa.


### Entrega:
El formato de entrega va a ser un notebook que contenga el análisis de los datos, las conclusiones a
las que se llegan a partir de ese análisis y finalmente un algoritmo simple (baseline) que intente
predecir el target (si el usuario va a ver Frozen 4 en el mismo cine o no).


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

### Explicación de los datos recolectados:
- **volvera**: (variable target) entero que representa 0: no volvería, 1: si volvería
- **tipo_de_sala**: El tipo de la sala (2d, 3d, 4d) [2d: sala común, 3d: sala 3D, 4d: sala 4D]
- **genero**: género con el cual el usuario se identifica en la encuesta
- **edad**: edad del usuario que completa la encuesta
- **amigos**: cantidad de amigos con los que fue a ver la película (Frozen 3)
- **parientes**: cantidad de familiares con los que fue a ver la película (Frozen 3)
- **ticket**: código del ticket
- **precio**: precio pagado por el ticket, en franjas de valor ordenadas de 1 a 50
- **fila**: fila dentro de la sala
- **cine**: nombre del cine [FiuMark tiene varias sedes]
- **nombre**: nombre del usuario que completa la encuesta

Nota: algunos campos en la encuesta **pueden estar vacíos**.

El link de los datos se encuentra en: https://drive.google.com/drive/folders/1RTD8HGy3YnTctcaz2b_xjKe8rJqNt7EV?usp=sharing


### Fecha de Entrega
- Entrega del notebook: Miércoles 28 de octubre. Al mail orga.datos.fiuba@gmail.com
- Defensa oral del tp: Martes 3 de noviembre.

### Condiciones de una entrega válida:
- El notebook debe poder ser ejecutado de forma secuencial de principio a fin por su corrector, todas las dependencias de librerías
deben estar en un requirements.txt.
- La función baseline debe llamarse baseline(X: pd.DataFrame) -> List[int].
la cual debe recibir un pandas dataframe producido de la lectura del archivo de testeo original y devolver una lista

