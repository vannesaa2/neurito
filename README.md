# Taller Neurito 
# Keldryn Vannesa Sanchez Castro 

# SUPER IMPORTANTE 
Para ejecutar los experimentos, correr "uv run cambiosParaExperimentos.py" en la powerShell 
Todos los números de este documento salen a partir de los cambios y experimentos que hice en "cambiosParaExperimentos" 


## Experimentos

### 1. Dos necesidades fijas, variar la tercera

Se corrio cada necesidad de 0 a 1 en pasos de 0.01 y se anota donde cambia la cosa 

1. Fijas: energía 0.5 (cansancio 0.5), aburrimiento 0.3 
    Vario: Hambre
    Cambio de accion: DORMIR > COMER en hambre = 0.50
2. Fijas: hambre 0.3, aburrimiento 0.3
    Vario: Energía
    Cambio de accion: DORMIR > COMER en energía = 0.7 (cansancio baja de 0.3)
3. Fijas: hambre 0.3, energía 0.5 (cansancio 0.5)
    Vario: Aburrimiento
    Cambio de accion: DORMIR > JUGAR en aburrimiento = 0.50
4. Fijas: ehambre 0.2, energía 0.9 (estado calmo) 
    Vario: Aburrimiento
    Cambio de accion: COMER > JUGAR en aburrimiento = 0.21

#### Que paso ? 
La accion cambia justo cuando la necesidad que varía supera a la mayor de las otras dos prioridades. Eso lo define (0.50, 0.30 de cansancio, 0.50 y 0.20 en los cuatro casos), y la red lo aprende bien. No es fijo: depende de las otras dos necesidades. 
Ejemplo: con hambre 0.2 y cansancio 0.1, basta un aburrimiento de 0.21 para pasar a JUGAR, mientras que con cansancio 0.5 hace falta más de 0.50


### 2. Prioridades casi iguales y Softmax

1. Caso 1
    - Hambre / Energía / Aburrimiento: 0.50 / 0.50 / 0.50
    - Prioridades ($H$, $1-E$, $A$): 0.50, 0.50, 0.50
    Softmax (COMER, DORMIR, JUGAR): 0.352, 0.334, 0.313
2. Caso 2
    - Hambre / Energía / Aburrimiento: 0.51 / 0.49 / 0.50
    - Prioridades ($H$, $1-E$, $A$): 0.51, 0.51, 0.50
    - Softmax (COMER, DORMIR, JUGAR): 0.373, 0.354, 0.273
3. Caso 3
    - Hambre / Energía / Aburrimiento: 0.52 / 0.50 / 0.48
    - Prioridades ($H$, $1-E$, $A$): 0.52, 0.50, 0.48
    - Softmax (COMER, DORMIR, JUGAR): 0.484, 0.318, 0.199
4. Caso 4
    - Hambre / Energía / Aburrimiento: 0.55 / 0.50 / 0.45
    - Prioridades ($H$, $1-E$, $A$): 0.55, 0.50, 0.45
    - Softmax (COMER, DORMIR, JUGAR): 0.664, 0.250, 0.085
5. Caso 5
    - Hambre / Energía / Aburrimiento: 0.60 / 0.50 / 040
    - Prioridades ($H$, $1-E$, $A$): 0.60, 0.50, 0.40
    - Softmax (COMER, DORMIR, JUGAR): 0.856, 0.128, 0.016
6. Caso 6
    - Hambre / Energía / Aburrimiento: 0.80 / 0.50 / 0.20
    - Prioridades ($H$, $1-E$, $A$): 0.80, 0.50, 0.20
    - Softmax (COMER, DORMIR, JUGAR): 0.996, 0.004, 0.000
7. Caso 7
    - Hambre / Energía / Aburrimiento: 0.90 / 0.10 / 0.10
    - Prioridades ($H$, $1-E$, $A$): 0.90, 0.90, 0.10
    - Softmax (COMER, DORMIR, JUGAR): 0.488, 0.512, 0.000

#### Que paso ? 

- Con tres prioridades empatadas la salida es casi uniforme (≈ 0.33 cada una): la red "duda".
- Una diferencia de apenas 0.02 en las prioridades ya mueve a COMER de 0.35 a 0.48; con 0.10 de diferencia supera 0.85. Softmax amplifica las diferencias
- En el empate exacto entre COMER y DORMIR (última fila) la red elige DORMIR con 51.2 % vs 48.8 %. elige COMER solo porque "argmax" desempata por el primer índice. No es un error "real": el empate no tiene respuesta correcta, y la red no tiene forma de saber la regla de desempate.
- Los porcentajes de Softmax son solo cuanto prefiere la red una salida frente a las otras.

### 3. Tamaño de la capa oculta (3, 6, 10, 20 neuronas)

Resultados con semilla 42 (misma inicialización de datos) y, para ver la variabilidad, promedio de 5 semillas (0–4). Todos con 1200 épocas y lr 0.1.

| Neuronas | Parámetros | Semilla 42: pérdida / prueba | Media 5 semillas: pérdida | Media 5 semillas: precisión de prueba |
|---|---|---|---|---|
| 3 | 24 | 0.1618 / 96.30 % | 0.2768 (± 0.1976) | 88.88 % (± 14.04 %) |
| 6 | 45 | 0.1225 / 99.40 % | 0.1534 (± 0.0133) | 97.78 % (± 0.78 %) |
| 10 | 73 | 0.1262 / 99.30 % | 0.1222 (± 0.0044) | 98.98 % (± 0.31 %) |
| 20 | 143 | 0.1220 / 98.80 % | 0.1207 (± 0.0033) | 99.18 % (± 0.23 %) |
 
Conclusión:

-3 neuronas: funcionan con suerte (96 % con la semilla 42) pero son inestables: con otras semillas la precisión cae mucho (la desviación es de 14 puntos). Una explicación probable (no la comprobé en el experimento) es que con tan pocas neuronas ReLU, si alguna "muere" (salida siempre 0), la red pierde capacidad para separar las tres regiones.
- De 6 a 10 neuronas hay mejora clara en estabilidad y pérdida; de 10 a 20 la mejora es mínima (0.1222 → 0.1207 en pérdida). Más neuronas no significa mejor resultado, hay rendimientos decrecientes. Para este problema (que es casi lineal por tramos) 6–10 neuronas son suficientes.
- Con una sola semilla (42) el orden de precisión de prueba es incluso engañoso (6 neuronas queda por encima de 20). Para comparar modelos conviene promediar varias ejecuciones.

### 4. Épocas y tasa de aprendizaje

Épocas (lr 0.1, 6 neuronas):

| Épocas | Pérdida | Entrenamiento | Prueba |
|---|---|---|---|
| 50 | 0.9388 | 61.50 % | 60.60 % |
| 200 | 0.4984 | 91.50 % | 91.80 % |
| 600 | 0.1944 | 98.93 % | 98.60 % |
| 1200 | 0.1225 | 99.57 % | 99.40 % |
| 3000 | 0.0711 | 99.70 % | 99.50 % |
| 10000 | 0.0368 | 99.77 % | 99.70 % |

La pérdida mejora siempre, pero con rendimientos decrecientes: casi toda la precisión se gana antes de la época 600. De 1200 a 10 000 épocas la precisión sube solo 0.3 puntos. No hay sobreajuste visible porque los datos son sintéticos, abundantes y la red es pequeña.

Tasa de aprendizaje (1200 épocas):

| lr | Pérdida final | Veces que la pérdida sube | Prueba | Comportamiento |
|---|---|---|---|---|
| 0.001 | 1.0333 | 0 | 41.80 % | **Se estanca** (demasiado lento) |
| 0.01 | 0.7376 | 0 | 82.90 % | Mejora, pero aún sin converger |
| 0.1 | 0.1225 | 0 | 99.40 % | Estable (valor del proyecto) |
| 0.5 | 0.0484 | 0 | 99.60 % | Estable y más rápido |
| 1 | 0.0447 | 545 | 98.70 % | Mejor pérdida, pero **empieza a oscilar** |
| 2 | 0.0557 (mín. 0.0531) | 582 | 97.30 % | Oscilante, termina peor que su mínimo |
| 5 | 1.0979 (mín. 0.9163) | 165 | 33.80 % | **Inestable**, colapsa |
| 20 | 6.1621 | 800 | 33.80 % | **Diverge** |
| 50 | 14.5763 | 737 | 34.20 % | **Diverge** |

Tres regímenes: lr muy pequeña → la red casi no aprende; lr moderada → converge suave; lr grande → los pasos se pasan del mínimo, la pérdida sube y baja (≥ 1) y, con valores aún mayores, la red acaba prediciendo casi siempre la misma clase (~34 %, el nivel de azar con tres clases balanceadas). Curiosamente, 0.5 resultó mejor que el 0.1 elegido en el proyecto; el límite de estabilidad está entre 0.5 y 2.

### 5. ¿Qué datos y etiquetas requeriría una cuarta acción?

Ejemplo: BAÑARSE (o ASEARSE), asociado a una nueva necesidad suciedad.

1. **Una cuarta entrada** si la acción responde a una necesidad nueva: `X` pasa de 3 a 4 columnas (hambre, energía, aburrimiento, suciedad) y `W1` de forma `(3, h)` a `(4, h)`.
2. **Una cuarta salida:** `W2` pasa de `(h, 3)` a `(h, 4)` y `b2` de `(1, 3)` a `(1, 4)`. Softmax ahora reparte entre 4 clases.
3. **Etiquetas nuevas:** la clase 3 = BAÑARSE. Con la política sintética actual sería `argmax([hambre, 1 − energía, aburrimiento, suciedad])`. Hay que definir además cómo se desempata.
4. **Datos de cada clase en cantidad suficiente.** Hay que generar ejemplos donde la suciedad sea la necesidad dominante; si la nueva acción es rara, la red casi no la verá y aprenderá a ignorarla (desbalance de clases). Conviene revisar `np.bincount(targets)`.
5. **Reentrenar desde cero** (o ampliar pesos con cuidado): los pesos viejos no sirven con otra forma de red. Después, repetir la evaluación con matriz de confusión.
6. **Cambios fuera de la red:** nuevas imágenes (`bañarse_1.png`, ...), tecla para modificar la suciedad, etiquetas de texto y revisar la regla de CALMA para incluir la nueva necesidad.
7. Si en el futuro las etiquetas vinieran de comportamiento real (por ejemplo, observaciones o votos de usuarios) y no de una fórmula, haría falta recopilar y etiquetar esos ejemplos, y revisar que no haya sesgos o contradicciones.



## Preguntas

### 1. ¿En qué se diferencian pesos y sesgos?

- Los **pesos** (`W1`, `W2`) multiplican las entradas de cada neurona y determinan **cuánta influencia** tiene cada entrada sobre la salida y en qué dirección (un peso negativo la invierte). Son la forma de la conexión entre neuronas.
- Los **sesgos** (`b1`, `b2`) se **suman** al resultado y desplazan el umbral de activación de la neurona: dicen cuánto debe valer la suma para que la neurona "se encienda", independientemente de las entradas. Sin sesgo, una neurona con entradas en cero siempre daría 0.

En una neurona: `salida = f(w1·x1 + w2·x2 + w3·x3 + b)`. Geométricamente, los pesos giran/inclinan la frontera de decisión y el sesgo la desplaza. En este proyecto los sesgos empiezan en 0 y los pesos en valores aleatorios pequeños (`randn * 0.5`); ambos se actualizan con descenso del gradiente.

### 2. ¿Por qué el cansancio es `1 - energía`?

Porque la política compara **urgencias** y las tres tienen que estar en la misma dirección y la misma escala [0, 1]. Hambre y aburrimiento son necesidades: más valor = más urgencia. La energía va al revés: más energía = menos necesidad de dormir. Con `1 − energía` obtenemos "cansancio", que sube cuando la energía baja, y así `argmax` puede comparar manzanas con manzanas. Sin la inversión, un Neurito con energía 1.0 (descansado) parecería tener la prioridad más alta de dormir.

La red nunca ve el "cansancio"; recibe la energía cruda y tiene que aprender la inversión por sí misma (con pesos negativos y sesgos). Eso lo logra bien, y es un buen ejemplo de qué hacen los pesos y los sesgos: `1 − x` es exactamente un peso de −1 con un sesgo de +1.

### 3. ¿Qué hace ReLU y qué hace Softmax?

- **ReLU** (`max(0, x)`) se aplica en la capa oculta. Deja pasar los valores positivos y convierte los negativos en 0. Es la **no linealidad** que permite que la red aprenda fronteras más complejas que una línea recta (sin ella, dos capas lineales equivaldrían a una sola). También es barata de calcular. Su riesgo es que una neurona que siempre recibe valores negativos deja de aprender ("neurona muerta"), lo que podría contribuir a la inestabilidad con 3 neuronas.
- **Softmax** se aplica en la salida. Convierte las tres puntuaciones en números entre 0 y 1 que **suman 1**, y amplifica las diferencias entre ellas (por la exponencial). Así se pueden leer como el "reparto de preferencia" entre COMER, DORMIR y JUGAR, y se puede usar con la pérdida de entropía cruzada. La acción elegida es la de mayor valor. En el código se resta el máximo antes de exponenciar para evitar desbordamientos numéricos.

### 4. ¿Por qué una precisión alta puede ser engañosa?

En este mismo proyecto, 99.40 % parece casi perfecto, pero:

1. **Los errores se concentran donde importa.** En la prueba hay solo 6 fallos de 1000 (4 DORMIR→JUGAR y 2 JUGAR→COMER), y los errores se concentran en la frontera: cuando las dos prioridades mayores difieren menos de 0.02, la precisión baja a 90 %.
2. **Solo mide lo que se parece a los datos de entrenamiento.** Con entradas fuera del rango [0, 1] (que la red nunca vio) la precisión cae a 94.2 %.
3. **Mide acierto contra una etiqueta inventada.** 99.4 % significa que la red imita bien la política sintética, no que tome buenas decisiones para una mascota real. Si la política fuera mala o incompleta, la precisión no lo detectaría.
4. **Ignora lo que la red no puede hacer.** La pantalla muestra CALMA (≈ 5.7 % de los casos de prueba caen en esa región) y la red no puede predecirla; esa parte queda fuera del número.
5. **El desbalance de clases la puede inflar.** Aquí el azar con la clase mayoritaria da 34 %, así que 99.4 % sí es mucho mejor; pero con una cuarta acción poco frecuente un modelo que la ignore podría seguir teniendo precisión alta.
6. **Un número global esconde los errores por clase y su costo.** Hacen falta matriz de confusión, precisión/recall por clase y pruebas en casos difíciles, no solo un porcentaje.
7. **Un solo resultado no basta.** Con 3 neuronas la misma arquitectura dio 96 % con una semilla y bastante menos en promedio; hay que repetir con varias semillas.
