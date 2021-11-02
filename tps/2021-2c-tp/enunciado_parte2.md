# Introducción
Luego de la presentación del informe y el baseline, Flint Lockwood quiere profundizar la capacidad de predicción. Gracias al éxito 
logrado en la primera campaña él tiene más confianza en ustedes y sus “algoritmos” y está ansiosa por probar 
las avanzadas técnicas de inteligencia artificial de las que todo el mundo habla.


## Tarea
Flint Lockwood está muy interesado en utilizar algoritmos de machine learning desde que escuchó 
que otros scientificos lo utilizan para pronosticar otros experimentos fallidos. Con su conocimiento al respecto exige 
que probemos varios modelos (al menos 5 tipos distintos) reportando cual de todos fue el mejor (según la métrica AUC-ROC), 
también pretende que utilicemos técnicas para buscar la mejor configuración de hiperparámetros, que intentemos hacer al 
menos un ensamble, que utilicemos cross-validation para comparar los modelos y que presentemos varias métricas del modelo final:  
- AUC-ROC
- Matriz de confusión
- Accuracy
- Precisión
- Recall

Flint también sabe los dilemas que resultan de llevar un prototipo a producción, por lo que nos pidió 
encarecidamente que dejemos muy explícitos los pasos de pre-procesamiento/feature engineering que usamos en cada 
modelo, y que dejemos toda la lógica del preprocesado en un archivo python llamado preprocesing.py en donde van a 
estar todas las funciones utilizadas para preprocesamiento, de hecho, él espera que apliquemos al menos dos tecnicas
de preprocesamiento distintos por cada tipo de modelo y espera que si dos modelos tienen el mismo preprocesado 
entonces usen la misma función en preprocessing.py.  

## Entrega
El formato de entrega va a ser un breve informe en PDF con:

**(TABLA 1)** Tabla que liste todos los pre-procesamientos utilizados, con el estilo:  
\<nombre preprocesamiento\> \< explicación simple\> \< nombre de la función de python \>  
En donde:  
- **nombre preprocesamiento:** es un nombre que ustedes elijan para representar lo que hace el preprocesado.
- **explicación simple:** es una descripción en no más de 2 líneas de la lógica de preprocesado.
- **nombre de la función de python:** nombre de la función de python que va a estar localizada en preprocessing.py


**(TABLA 2)** Tabla que liste:  
\<Nombre Modelo\> \<nombre preprocesamiento\>  \<AUC-ROC\> \<Accuracy\> \<Precision\> \<Recall\> \<F1 score\>  
En donde :  
- **Nombre Modelo:** es el mejor modelo de los de su mismo tipo, enumerados en orden secuencial que se fueron realizando (\<número\> - \<nombre\>).
- **nombre preprocesamiento:** es el nombre del preprocesamiento (tiene que estar presente en la tabla anterior)
- (el resto de las columnas son las métricas de ese Nombre Modelo)

En concordancia con lo anterior, se espera que cada Nombre Modelo este en un notebook separado con el nombre
\<Nombre Modelo\>.ipynb y que dentro del mismo esté de forma clara la llamada a los preprocesados, su entrenamiento, 
la evaluación del mismo y finalmente una predicción en formato csv de un archivo nuevo localizado
en: https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE  
El director nos pide que por cada modelo listado en la tabla, hagamos las predicciones de este archivo y en la entrega junto con los notebook 
también entreguemos todas las predicciones. El nombre del archivo con las predicciones tiene que ser \<Nombre Modelo\>.csv, 
el tiene pensado chequear las métricas de estas predicciones minutos antes de pagarnos ( esperemos que no nos cague  👀 ).

**(CONCLUSIÓN)**
Finalmente luego de poner las tablas TABLA 1 y TABLA 2, nos piden que lleguemos a una conclusión sobre qué modelo 
recomendamos y por qué y que lo comparemos con respecto al baseline que anteriormente implementamos. Tambien quiere 
que agreguemos un pequeño analisis de que modelo elegiriamos si se necesitase tener la menor cantidad de falsos positivos
o si necesitan tener una lista de todos los que potencialmente son de valor adquisitivo sin preocuparse demasiado 
si metemos en la misma personas que realmente no tienen alto valor adquisitivo.


## Notas Técnicas:
- El formato esperado para las predicciones realizadas en cada .csv es igual al del archivo de ejemplo https://docs.google.com/spreadsheets/d/1jc4bfOyp80opnBnTBupqXnJajyF3a9NVuS9_c8XR7zU en donde por cada 
línea del archivo se tiene:  
\<id\> \<tiene_alto_valor_adquisitivo\>  
- Todos los notebooks deben poder ser ejecutados de principio a fin por su corrector produciendo los mismos resultados
- Todas las dependencias de librerías deben estar en un requirements.txt
- La entrega se tiene que realizar en el mismo repositorio de la primera entrega, en una carpeta llamada parte_2
- Las predicciones de cada modelo se deberan guardar en el directorio parte_2/predicciones

## Fecha de entrega:
- Entrega codigo: 8 de Diciembre (inclusive, hasta las 23:59)
- Defensa oral del tp: 14 de Diciembre (misma modalidad por turnos de la anterior parte)

## Forma de entrega:
- Deberan cargar todo el trabajo en git y solo re tomará como valido los cambios subidos hasta la fecha de entrega mencionada anteriormente.

## Consideraciones sobre uso de Git:
(disclaimer: pongo está seccion a modo abodago por hay cada uno que me viene con algunas cosas que no se pueden creer)
- Todo progreso que se vaya realizando debera ser subido a git, no se considerará una entrega valida si suben el codigo pocos dias antes de la fecha de entrega (mas precisamente si solo suben el codigo 2 dias antes de la fecha de entrega).
- TODOS deberan realizar aportes al git, si un alumno no realiza aporte alguno este perdera automaticamente la cursada.
- Se podra usar el historial de commits para validar o no si un alumno participo en la elaboracion del tp y si el corrector lo considera, podria perder la cursada por no haber aportado significativamente al mismo (nota: no esperamos que sea un 50%-50% perfecto, pero vamos a mirar que no sea una distribucion muy desigual).

Nota: entre la entrega y la defensa oral se pueden llegar a pedir correcciones para antes de la defensa oral de 
considerarse necesarias.
