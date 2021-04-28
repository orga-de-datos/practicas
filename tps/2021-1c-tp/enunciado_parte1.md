# 2021-1C TP
## FIUFIP: Que No Se Escape Nadie (?)

### Introducción:
Una agencia gubernamental de recaudación de impuestos la FIUFIP nos a pedido que atrapemos a malechores 
que evaden impuestos y destruyen la economía.

Para ello la agencia desea entender la información que han estado recolectando de gente que paga los impuestos y quiere
tener un sistema para el cual saber si una persona tiene altos ingresos o bajos ingresos a partir de los mismos.

Estos datos consisten en una serie de atributos de la persona que la agencia fue llenando y catalogando a la persona 
como de altos ingresos o no.

La agencia quiere utilizar la información recolectada para dirigir campañas de recaudación de impuestos y poder dirigir
a los fiuagentes recaudadores a inspeccionar.

### Tarea:

La primer tarea que nos piden es que presentemos un informe explicando qué conclusiones se pueden
llegar a partir de analizar los datos recolectados. La agencia quiere saber cómo entender cada
uno de los datos, cómo se relacionan entre sí y si se puede sacar alguna conclusión o descubrir
un patrón a partir de estos. Adicionalmente quieren saber cuáles son los factores más importantes
que determinan si un usuario tiene altos o bajos ingresos.


Si bien la agencia quiere empezar a usar técnicas avanzadas de predicción e inteligencia artificial,
todavía tiene desconfianza en las mismas (personalmente comentaron que tienen miedo que las máquinas
se revelen contra ellos) por lo que inicialmente no quieren nada complicado sino una serie muy simple
de decisiones lógicas que les permitan en poco tiempo hacer una primera ronda de campaña digital. Se
espera que este código simple (baseline) tenga una accuracy aceptable (mayor a ??%) y que esté basada y justificada
en la investigación previa.


### Entrega:
El formato de entrega va a ser un notebook que contenga el análisis de los datos, las conclusiones a
las que se llegan a partir de ese análisis y finalmente un algoritmo simple (baseline) que intente
predecir el target (si el usuario tiene altos ingresos o bajos ingresos).


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
   
Se espera que presenten el notebook en una charla de 20 minutos y dijeron que quieren explícitamente el formato:
- pregunta
- gráfico/s para responder esa pregunta
- por cada gráfico quieren un comentario escrito de que se interpreta en el mismo
- respuesta a la pregunta en base a todos los gráficos y análisis de los mismos

### Explicación de los datos recolectados:

**edad**: número, indica la edad   
**categoria_de_trabajo**: texto, cada valor indica el tipo de trabajo  
**educacion_alcanzada**: texto, cada valor indica la educación alcanzada  
**anios_estudiados**: número, indica la cantidad de años que estudió  
**estado_marital**: texto, indica el estado marital  
**trabajo**: texto, indica que tipo de trabajo realiza  
**rol_familiar_registrado**: texto, indica que rol tiene dentro del grupo familiar  
**religion**: texto, indica a que religión pertenece  
**genero**: texto, indica género  
**horas_trabajo_registradas**: número, indica la cantidad de horas que trabaja por semana  
**barrio**: texto, indica que barrio de capital reside  
**tiene_alto_valor_adquisitivo**: número (variable target) indica si tiene alto valor adquisitivo (valor 1) o bajo
valor adquisitivo (valor 0)  
**ganancia_perdida_declarada_bolsa_argentina**: número, indica el resultado de las operaciones 
en bolsa que realizó durante el último año.  

Nota1: algunos campos en la encuesta **pueden estar vacíos**.  
Nota2: San Isidro es un valor valido en el campo barrio  
Nota3: el valor de orden en el campo educacion_alcanzada es x_grado < x_anio < universidad_x_anio  

El link de los datos se encuentra en: https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/edit?usp=sharing


### Fecha de Entrega
- Entrega del notebook: Miercoles 12 de mayo. En un repo de github PRIVADO compartido al mail orga.datos.fiuba@gmail.com
- Defensa oral del TP: Martes 18 de mayo.

### Condiciones de una entrega válida:
- El notebook debe poder ser ejecutado de forma secuencial de principio a fin por su corrector, todas las dependencias 
  de librerías deben estar en un requirements.txt.
- La función baseline debe llamarse baseline(X: pd.DataFrame) -> List[int].
la cual debe recibir un pandas dataframe producido de la lectura del archivo de testeo original y devolver una lista
  con las predicciones (1 para si es de altos ingresos, 0 para si es de bajos ingresos)

