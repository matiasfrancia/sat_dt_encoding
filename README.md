# Tarea 2 Lógica Computacional Aplicada

Implementación del paper Learning Optimal Decision Trees with SAT [1]
Para poder ejecutar este código es necesario contar con la librería PySAT y sus dependencias.

## Pasos
1. Instalar PySAT
2. Abrir archivo main.ipynb
3. Ejecutar celdas del archivo cambiando los comandos de consola según la necesidad
4. Revisar árbol generado en la carpeta binarytree/cnf/

Notas: 
- El código principal se encuentra en learning_dt_with_sat.py, y es llamado por main.ipynb
- Se debe escoger el número de nodos del árbol a través del parámetro ```--size``` al llamar a learning_dt_with_sat.py
- En el archivo principal hay una función llamada generate_constraints, la cual genera las restricciones del encoding. Si se comentan las llamadas a las funciones self.add_discrimination_for_value_0(N), self.add_discrimination_for_value_1(N) y self.add_path_activation_constraint(N, K), el código corre hasta el final y genera un árbol de decisión binario. Sin embargo, no codifica las restricciones asociadas a la correcta clasificación de las muestras.

## Referencias

[1] Narodytska, N., Ignatiev, A., Pereira, F., & Marques-Silva, J. (2018). Learning optimal decision trees with SAT. In International Joint Conference on Artificial Intelligence 2018 (pp. 1362-1368). Association for the Advancement of Artificial Intelligence (AAAI).