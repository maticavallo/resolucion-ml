Explicación de mi practica de Machine Learning - Airbnb - MATIAS CAVALLO

Básicamente hay que predecir el precio de alojamientos de Airbnb. Como el precio es un número, es un problema de regresión.  Por lo tanto, me base en en el ejemplo de las casas de King County, que fue el que más me sirvió como guía.

Lo primero fue ver qué columnas usar, porque muchas eran cosas como IDs, URLs, descripciones, en fin, cosas que no sirven para predecir un precio. Entonces me quedé con las que tienen logica, como saber cuántas habitaciones tiene, en qué ciudad está, las puntuaciones de reviews, etc.
Aparte, el precio venía como string, entonces hice una conversion a float. 

Había mucha data missing como NaN. Pensé en borrar esas filas pero iba a perder muchos datos. Entonces los rellené con determinada logica, para cosas como bathrooms y bedrooms usé la mediana. porque si hay algún valor muy raro (tipo uno con 10 baños), la mediana no se altera tanto. Para las puntuaciones de reviews usé la media porque ahí tiene más sentido y para strings como Property Type o City, usé la moda.

Vi que había algunos precios muy altos, a lo mejor oferta de lujo o directamente errores. Esos valores raros confunden al modelo, entonces saqué todo lo que estaba por encima del 99% para evitar que valores extremos distorsionen el entrenamiento y solo y solo pierdo el 1%.

Hice histogramas del precio para ver cómo se distribuía. También hice scatter plots para ver si a más habitaciones más precio, y lo mismo con otras variables como capacidad y reviews

Lo que más me sirvió fue la matriz de correlación. Ahí vi que Accommodates (capacidad) tiene mucha correlación con el precio, lo cual tiene sentido. Bedrooms también. Pero ojo, las review scores tienen poca correlación, o sea que una buena puntuación no necesariamente significa que sea un precio mas alto.

Tenía strings en Property Type  y las tuve que convertir a números porque los modelos solo entienden números usando LabelEncoder. Usé 80% para entrenar y 20% para probar y puse shuffle=True para que mezcle los datos antes de dividir, así no pasa que en train quedan todas las casas de una ciudad y en test las de otra.

Arranqué con regresion lineal porque es el más simple y me parecio mas logico empezar por lo mas simple. De todas formas, yo me intuia que no iba a ser el mejor porque el precio no depende de forma perfectamente lineal de las variables.

Despues segui con Lasso porque al ser una regresion Lineal pero con regularización, o sea un modo de freno para no sobreajustar, poniendo por ejemplo  algunas variables en cero y las elimina si no sirven. Lo que hice fue usar GridSearchCV para encontrar el mejor valor de alpha como parametro para controlar cuanto freno poner. Probé 20 valores diferentes con validación cruzada de 3 fold.

Luego, con la tecnica de arbol de regularizacion la probe porque esto puede encontrar relaciones que no son líneas rectas, aunque hace mucho overfitting, por eso puse max_depth a 10 para limitarlo.

Para terminar, usé Random Forest, que imaginaba que iba a ser el mejorcito, porque funciona como un conjunto de muchos árboles que trabajan juntos y promedian sus resultados. Puse n_estimators = 100 (es decir, 100 árboles), max_depth = 15 para limitar su profundidad y n_jobs = -1 para que use todos los procesadores y vaya más rápido. Además, Random Forest permite ver qué variables son más importantes, lo cual está bueno para entender qué es lo que realmente influye en el precio.

Evalué todos los modelos usando MSE, RMSE y R2.
      El MSE (Mean Squared Error) es el error cuadrático medio, o sea el promedio de las diferencias al cuadrado entre los valores reales y los predichos.
      El RMSE es la raíz del MSE, y me gusta más porque está en las mismas unidades que el precio (dólares). Si me da 50, significa que en promedio me equivoco unos 50 dólares; cuanto más bajo, mejor.
      El R2 es un número entre 0 y 1: cuanto más se acerca a 1, mejor predice el modelo. A partir de 0.7 ya se considera bastante bueno, y si llega a 0.8 o más, excelente.

CONCLUSION.

Como conclusion puedo decir  que Random Forest fue el mejor por lejos, ya que tuvo el RMSE más bajo y el R2 más alto en el conjunto de test. En promedio, las predicciones se equivocan en unos 30 dólares, y el precio se explicaria en un 64% por las variables importantes y objetivas que contamos y que aparte tienen logica (ubicaciones, habitaciones, baños). El 36% restante probablemente se debe a factores que no están en el dataset, como la calidad de las fotos o la temporada del año como cosas que se me ocurren. La ubicación sí es un factor relevante, aunque el tamaño físico del alojamiento tiene mayor peso en la predicción del precio. Aparte esta experiencia me demostro que limpiar bien los datos es muy importante, más de lo que pensaba. Y entiendo que Random Forest funciona mejor porque porque puede combinar muchos árboles para tomar mejores decisiones. 

Desde ya muchas gracias por todo. Saludos.